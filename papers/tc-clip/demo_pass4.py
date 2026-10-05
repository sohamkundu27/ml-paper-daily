#!/usr/bin/env python3
"""End-to-end demo for TC-CLIP Pass 4 on synthetic action recognition task."""

import torch
import torch.nn as nn
import torch.optim as optim
from tc_clip import TCClipPass4, ToyActionDataset


def train_pass4_demo():
    """Train TC-CLIP Pass 4 on synthetic action recognition task."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    num_actions = 5
    feature_dim = 256
    batch_size = 4
    num_epochs = 3
    learning_rate = 0.001

    print(f"{'='*60}")
    print("TC-CLIP Pass 4: End-to-End Action Recognition Demo")
    print(f"{'='*60}\n")

    print("Configuration:")
    print(f"  - Number of action classes: {num_actions}")
    print(f"  - Feature dimension: {feature_dim}")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Epochs: {num_epochs}")
    print(f"  - Device: {device}\n")

    dataset = ToyActionDataset(
        num_samples=100,
        num_actions=num_actions,
        num_frames=8,
        height=64,
        width=64,
        channels=3,
        seed=42,
    )

    model = TCClipPass4(
        feature_dim=feature_dim,
        freeze_backbone=False,
        aggregation="mean",
        num_context_tokens=4,
        num_heads=4,
        vocab_size=5000,
        num_action_classes=num_actions,
    ).to(device)

    trainable_params = model.get_learnable_parameters()
    print(f"Trainable parameters: {sum(p.numel() for p in trainable_params):,}\n")

    optimizer = optim.Adam(trainable_params, lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    print("Training Loop:")
    print(f"{'Epoch':<8} {'Batch':<8} {'Loss':<12} {'Accuracy':<12}")
    print("-" * 50)

    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0
        epoch_correct = 0
        epoch_total = 0

        for batch_idx in range(10):
            videos, labels = dataset.get_batch(batch_size=batch_size)
            videos = videos.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            output = model(videos)
            logits = output["action_logits"]
            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()

            _, predictions = logits.max(dim=1)
            correct = predictions.eq(labels).sum().item()
            epoch_correct += correct
            epoch_total += batch_size
            epoch_loss += loss.item()

            if batch_idx % 5 == 0:
                batch_acc = correct / batch_size
                print(
                    f"{epoch + 1:<8} {batch_idx:<8} {loss.item():<12.4f} {batch_acc:<12.2%}"
                )

        avg_loss = epoch_loss / 10
        avg_acc = epoch_correct / epoch_total
        print(
            f"{'Epoch ' + str(epoch + 1) + ' Average':<8} "
            f"{'':<8} {avg_loss:<12.4f} {avg_acc:<12.2%}\n"
        )

    print(f"{'='*60}")
    print("Evaluation on Held-Out Synthetic Data")
    print(f"{'='*60}\n")

    model.eval()
    eval_correct = 0
    eval_total = 0

    print("Per-Action Accuracy:")
    print(f"{'Action':<15} {'Samples':<15} {'Accuracy':<15}")
    print("-" * 50)

    for action_id in range(num_actions):
        action_correct = 0
        action_total = 0

        for _ in range(5):
            videos, labels = dataset.get_batch(batch_size=4, action_id=action_id)
            videos = videos.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                output = model(videos)
                logits = output["action_logits"]

            _, predictions = logits.max(dim=1)
            action_correct += predictions.eq(labels).sum().item()
            action_total += labels.size(0)

        action_acc = action_correct / action_total if action_total > 0 else 0
        print(f"{'Action ' + str(action_id):<15} {action_total:<15} {action_acc:<15.2%}")
        eval_correct += action_correct
        eval_total += action_total

    overall_acc = eval_correct / eval_total if eval_total > 0 else 0
    print("-" * 50)
    print(f"{'Overall':<15} {eval_total:<15} {overall_acc:<15.2%}\n")

    print(f"{'='*60}")
    print("Sample Predictions")
    print(f"{'='*60}\n")

    sample_videos, sample_labels = dataset.get_batch(batch_size=3)
    sample_videos = sample_videos.to(device)
    sample_labels = sample_labels.to(device)

    with torch.no_grad():
        predictions, confidences = model.predict(sample_videos)

    action_names = [
        "Horizontal Motion",
        "Vertical Motion",
        "Diagonal Motion",
        "Circular Motion",
        "Static with Color",
    ]

    print(f"{'Sample':<10} {'True Action':<25} {'Predicted':<25} {'Confidence':<12}")
    print("-" * 75)

    for i in range(3):
        true_action = action_names[sample_labels[i].item()]
        pred_action = action_names[predictions[i].item()]
        confidence = confidences[i].item()
        match = "✓" if predictions[i].item() == sample_labels[i].item() else "✗"
        print(
            f"{i + 1:<10} {true_action:<25} {pred_action:<25} "
            f"{confidence:<12.2%} {match}"
        )

    print(f"\n{'='*60}")
    print("✅ Pass 4 Demo Complete")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    train_pass4_demo()
