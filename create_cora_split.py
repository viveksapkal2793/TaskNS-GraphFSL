import pickle
import torch_geometric
import os

# Load CoraFull dataset - path is relative to script location
cora_full = torch_geometric.datasets.CitationFull('./dataset', 'cora')
labels = cora_full.data.y
class_list = labels.unique().tolist()

print(f"Total classes in CoraFull: {len(class_list)}")

# Count nodes per class
nodes_per_class = {}
for cls in class_list:
    nodes_per_class[cls] = (labels == cls).sum().item()

print(f"\nNodes per class statistics:")
print(f"Min: {min(nodes_per_class.values())}")
print(f"Max: {max(nodes_per_class.values())}")
print(f"Mean: {sum(nodes_per_class.values()) / len(nodes_per_class):.2f}")

# Filter classes with at least minimum required nodes
# For aux sampling: need at least aux_num_per_way (20) nodes
# For ID sampling: need at least k_shot + n_query (3 + 15 = 18) nodes
min_nodes_required = 20  # max(20, 18)
valid_classes = [cls for cls, count in nodes_per_class.items() if count >= min_nodes_required]

print(f"\nClasses with >= {min_nodes_required} nodes: {len(valid_classes)} / {len(class_list)}")

if len(valid_classes) < 70:
    print(f"Warning: Only {len(valid_classes)} classes have enough nodes. Adjusting split accordingly.")
    class_list = valid_classes

# Create train/valid/test split according to paper: 25/20/25
num_classes = len(class_list)
num_train = 25
num_valid = 20
num_test = 25

if num_classes < num_train + num_valid + num_test:
    # Fallback proportional split if not enough classes
    num_train = int(0.36 * num_classes)  # ~25/70
    num_valid = int(0.29 * num_classes)  # ~20/70
    num_test = num_classes - num_train - num_valid
    print(f"\nAdjusted split to {num_train}/{num_valid}/{num_test}")

import random
random.seed(1234)  # Use same seed as in train.py for reproducibility
random.shuffle(class_list)

class_list_train = class_list[:num_train]
class_list_valid = class_list[num_train:num_train + num_valid]
class_list_test = class_list[num_train + num_valid:num_train + num_valid + num_test]

print(f"\nTrain classes: {len(class_list_train)}")
print(f"Valid classes: {len(class_list_valid)}")
print(f"Test classes: {len(class_list_test)}")

# Verify each class has enough nodes
print("\nVerifying class node counts:")
for split_name, split_classes in [("Train", class_list_train), ("Valid", class_list_valid), ("Test", class_list_test)]:
    min_count = min(nodes_per_class[cls] for cls in split_classes)
    print(f"{split_name}: min nodes = {min_count}")

# Create directory if it doesn't exist
os.makedirs('./dataset/cora', exist_ok=True)

# Save the split
with open('./dataset/cora/cls_split.pkl', 'wb') as f:
    pickle.dump((class_list_train, class_list_valid, class_list_test), f)

print("\nSaved cls_split.pkl successfully!")
print(f"File saved at: {os.path.abspath('./dataset/cora/cls_split.pkl')}")