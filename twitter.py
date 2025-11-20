import copy
import torch
import random
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score
from tqdm import tqdm

from dataset import FeatureDataset
from twittermodel import SimilarityModule, DetectionModule

# ---------------- CONFIG ----------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", DEVICE)

NUM_WORKER = 1
BATCH_SIZE = 64
LR = 1e-3
L2 = 0
NUM_EPOCH = 5   # keep small for tests


# ---------------- SEED ----------------
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


setup_seed(1)


# ---------------- DATA PREP ----------------
def prepare_data(text, image, label):
    nr_index = [i for i, l in enumerate(label) if l == 1]
    text_nr = text[nr_index]
    image_nr = image[nr_index]

    fixed_text = copy.deepcopy(text_nr)
    matched_image = copy.deepcopy(image_nr)
    unmatched_image = copy.deepcopy(image_nr).roll(shifts=3, dims=0)

    return fixed_text, matched_image, unmatched_image


# ---------------- TRAIN ----------------
def train():
    device = torch.device(DEVICE)

    dataset_dir = "data/twitter"
    train_set = FeatureDataset(
        f"{dataset_dir}/train_text_with_label.npz",
        f"{dataset_dir}/train_image_with_label.npz"
    )
    test_set = FeatureDataset(
        f"{dataset_dir}/test_text_with_label.npz",
        f"{dataset_dir}/test_image_with_label.npz"
    )

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

    similarity_module = SimilarityModule().to(device)
    detection_module = DetectionModule().to(device)

    loss_func_similarity = torch.nn.CosineEmbeddingLoss()
    loss_func_detection = torch.nn.CrossEntropyLoss()

    optim_task_similarity = torch.optim.Adam(similarity_module.parameters(), lr=LR)
    optim_task_detection = torch.optim.Adam(detection_module.parameters(), lr=LR)

    best_acc = 0

    for epoch in range(NUM_EPOCH):

        similarity_module.train()
        detection_module.train()

        total_sim_loss = 0
        total_det_loss = 0
        sim_correct = 0
        det_correct = 0
        sim_count = 0
        det_count = 0

        for step, (text, image, label) in tqdm(enumerate(train_loader)):

            text = text.to(device)
            image = image.to(device)
            label = label.to(device)

            # -------- TASK 1 (Similarity) --------
            fixed_text, matched_image, unmatched_image = prepare_data(text, image, label)
            fixed_text = fixed_text.to(device)
            matched_image = matched_image.to(device)
            unmatched_image = unmatched_image.to(device)

            t_match, i_match, pred_match = similarity_module(fixed_text, matched_image)
            t_unmatch, i_unmatch, pred_unmatch = similarity_module(fixed_text, unmatched_image)

            sim_pred = torch.cat([
                pred_match.argmax(1),
                pred_unmatch.argmax(1)
            ])

            sim_label0 = torch.cat([
                torch.ones(pred_match.size(0)),
                torch.zeros(pred_unmatch.size(0))
            ]).to(device)

            sim_label1 = torch.cat([
                torch.ones(pred_match.size(0)),
                -1 * torch.ones(pred_unmatch.size(0))
            ]).to(device)

            t_cat = torch.cat([t_match, t_unmatch])
            i_cat = torch.cat([i_match, i_unmatch])

            loss_sim = loss_func_similarity(t_cat, i_cat, sim_label1)

            optim_task_similarity.zero_grad()
            loss_sim.backward()
            optim_task_similarity.step()

            sim_correct += (sim_pred == sim_label0).sum().item()
            sim_count += sim_label0.size(0)

            total_sim_loss += loss_sim.item() * sim_label0.size(0)

            # -------- TASK 2 (Detection) --------
            t_align, i_align, _ = similarity_module(text, image)
            det_pred = detection_module(text, image, t_align, i_align)

            loss_det = loss_func_detection(det_pred, label)

            optim_task_detection.zero_grad()
            loss_det.backward()
            optim_task_detection.step()

            det_pred_label = det_pred.argmax(1)
            det_correct += (det_pred_label == label).sum().item()
            det_count += label.size(0)

            total_det_loss += loss_det.item() * label.size(0)

        # Compute training metrics
        train_sim_acc = sim_correct / sim_count
        train_det_acc = det_correct / det_count
        train_sim_loss = total_sim_loss / sim_count
        train_det_loss = total_det_loss / det_count

        # -------- TEST --------
        sim_acc_test, det_acc_test, sim_loss_test, det_loss_test, cm_sim, cm_det, \
            det_prec, det_rec, det_f1 = test(similarity_module, detection_module, test_loader)

        if det_acc_test > best_acc:
            best_acc = det_acc_test

        print("\n==================== EPOCH", epoch + 1, "====================")
        print("TASK 1 SIMILARITY:")
        print(f"Train Acc: {train_sim_acc:.3f} | Test Acc: {sim_acc_test:.3f}")
        print(f"Train Loss: {train_sim_loss:.3f} | Test Loss: {sim_loss_test:.3f}")
        print("Confusion Matrix:\n", cm_sim)

        print("\nTASK 2 DETECTION:")
        print(f"Train Acc: {train_det_acc:.3f} | Test Acc: {det_acc_test:.3f} | Best: {best_acc:.3f}")
        print(f"Train Loss: {train_det_loss:.3f} | Test Loss: {det_loss_test:.3f}")
        print("Confusion Matrix:\n", cm_det)
        print(f"Precision: {det_prec:.3f} | Recall: {det_rec:.3f} | F1: {det_f1:.3f}")
        print("============================================================\n")


# ---------------- TEST ----------------
def test(similarity_module, detection_module, test_loader):

    device = torch.device(DEVICE)
    similarity_module.eval()
    detection_module.eval()

    loss_sim_total = 0
    loss_det_total = 0
    sim_count = 0
    det_count = 0

    sim_labels_all = []
    sim_preds_all = []

    det_labels_all = []
    det_preds_all = []

    loss_func_similarity = torch.nn.CosineEmbeddingLoss()
    loss_func_detection = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for text, image, label in test_loader:
            text = text.to(device)
            image = image.to(device)
            label = label.to(device)

            fixed_text, matched_image, unmatched_image = prepare_data(text, image, label)
            fixed_text = fixed_text.to(device)
            matched_image = matched_image.to(device)
            unmatched_image = unmatched_image.to(device)

            # TASK 1 SIMILARITY
            t_match, i_match, p_match = similarity_module(fixed_text, matched_image)
            t_unmatch, i_unmatch, p_unmatch = similarity_module(fixed_text, unmatched_image)

            sim_pred = torch.cat([p_match.argmax(1), p_unmatch.argmax(1)])
            sim_label0 = torch.cat([
                torch.ones(p_match.size(0)),
                torch.zeros(p_unmatch.size(0))
            ]).to(device)

            sim_labels_all.append(sim_label0.cpu().numpy())
            sim_preds_all.append(sim_pred.cpu().numpy())

            t_cat = torch.cat([t_match, t_unmatch])
            i_cat = torch.cat([i_match, i_unmatch])
            sim_label1 = torch.cat([
                torch.ones(p_match.size(0)),
                -1 * torch.ones(p_unmatch.size(0))
            ]).to(device)

            loss_sim = loss_func_similarity(t_cat, i_cat, sim_label1)
            loss_sim_total += loss_sim.item() * sim_label0.size(0)
            sim_count += sim_label0.size(0)

            # TASK 2 DETECTION
            t_align, i_align, _ = similarity_module(text, image)
            det_pred = detection_module(text, image, t_align, i_align)

            det_pred_label = det_pred.argmax(1)

            det_labels_all.append(label.cpu().numpy())
            det_preds_all.append(det_pred_label.cpu().numpy())

            loss_det = loss_func_detection(det_pred, label)
            loss_det_total += loss_det.item() * label.size(0)
            det_count += label.size(0)

    # --- Combine all results ---
    sim_labels_all = np.concatenate(sim_labels_all)
    sim_preds_all = np.concatenate(sim_preds_all)

    det_labels_all = np.concatenate(det_labels_all)
    det_preds_all = np.concatenate(det_preds_all)

    # --- Metrics ---
    sim_acc = accuracy_score(sim_labels_all, sim_preds_all)
    det_acc = accuracy_score(det_labels_all, det_preds_all)

    cm_sim = confusion_matrix(sim_labels_all, sim_preds_all)
    cm_det = confusion_matrix(det_labels_all, det_preds_all)

    det_prec = precision_score(det_labels_all, det_preds_all, zero_division=0)
    det_rec = recall_score(det_labels_all, det_preds_all, zero_division=0)
    det_f1 = f1_score(det_labels_all, det_preds_all, zero_division=0)

    return (
        sim_acc, det_acc,
        loss_sim_total / sim_count,
        loss_det_total / det_count,
        cm_sim, cm_det,
        det_prec, det_rec, det_f1
    )


# ---------------- RUN ----------------
if __name__ == "__main__":
    train()
