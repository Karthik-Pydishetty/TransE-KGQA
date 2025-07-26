from collections import Counter
from torch.utils import data
from typing import Dict, Tuple

Mapping = Dict[str, int]


def create_mappings(dataset_path: str) -> Tuple[Mapping, Mapping]:
    """Creates separate mappings to indices for entities and relations."""
    # counters to have entities/relations sorted from most frequent
    entity_counter = Counter()
    relation_counter = Counter()
    with open(dataset_path, "r", encoding='utf-8') as f:
        for line_num, line in enumerate(f):
            # Strip whitespace and skip empty lines
            line = line.strip()
            if not line:
                continue
                
            # Split by tab and validate format
            parts = line.split("\t")
            if len(parts) != 3:
                print(f"Warning: Skipping malformed line {line_num + 1}: '{line}' (expected 3 parts, got {len(parts)})")
                continue
                
            head, relation, tail = parts
            # Skip if any part is empty
            if not head or not relation or not tail:
                print(f"Warning: Skipping line {line_num + 1} with empty components: '{line}'")
                continue
                
            entity_counter.update([head, tail])
            relation_counter.update([relation])
            
    entity2id = {}
    relation2id = {}
    for idx, (mid, _) in enumerate(entity_counter.most_common()):
        entity2id[mid] = idx
    for idx, (relation, _) in enumerate(relation_counter.most_common()):
        relation2id[relation] = idx
    return entity2id, relation2id


class FB15KDataset(data.Dataset):
    """Dataset implementation for handling FB15K and FB15K-237."""

    def __init__(self, data_path: str, entity2id: Mapping, relation2id: Mapping):
        self.entity2id = entity2id
        self.relation2id = relation2id
        self.data = []
        
        with open(data_path, "r", encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                # Strip whitespace and skip empty lines
                line = line.strip()
                if not line:
                    continue
                    
                # Split by tab and validate format
                parts = line.split("\t")
                if len(parts) != 3:
                    print(f"Warning: Skipping malformed line {line_num + 1} in dataset: '{line}'")
                    continue
                    
                head, relation, tail = parts
                # Skip if any part is empty
                if not head or not relation or not tail:
                    print(f"Warning: Skipping line {line_num + 1} with empty components in dataset: '{line}'")
                    continue
                    
                self.data.append([head, relation, tail])

    def __len__(self):
        """Denotes the total number of samples."""
        return len(self.data)

    def __getitem__(self, index):
        """Returns (head id, relation id, tail id)."""
        head, relation, tail = self.data[index]
        head_id = self._to_idx(head, self.entity2id)
        relation_id = self._to_idx(relation, self.relation2id)
        tail_id = self._to_idx(tail, self.entity2id)
        return head_id, relation_id, tail_id

    @staticmethod
    def _to_idx(key: str, mapping: Mapping) -> int:
        try:
            return mapping[key]
        except KeyError:
            return len(mapping)
