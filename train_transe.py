#!/usr/bin/env python3
"""
TransE Training Pipeline for LC-QuAD Knowledge Graph
Phase B: Training TransE with extracted triples
"""

import os
import json
import torch
import torch.optim as optim
from torch.utils import data as torch_data
from torch.utils import tensorboard
import numpy as np
from typing import Tuple, Dict, Any
import argparse
from sklearn.model_selection import train_test_split

# Import existing modules
import data
import metric
import model as model_definition
import storage
from config import Config

class LCQuADDataPreprocessor:
    """Handles splitting and preparing LC-QuAD triples for TransE training."""
    
    def __init__(self, triples_file: str, output_dir: str, test_size: float = 0.2, val_size: float = 0.1):
        self.triples_file = triples_file
        self.output_dir = output_dir
        self.test_size = test_size
        self.val_size = val_size
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
    def split_triples(self) -> Tuple[str, str, str]:
        """Split triples into train/validation/test sets."""
        print("Loading triples from:", self.triples_file)
        
        # Read all triples
        triples = []
        with open(self.triples_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    triples.append(line)
        
        print(f"Loaded {len(triples)} triples")
        
        # First split: separate test set
        train_val_triples, test_triples = train_test_split(
            triples, test_size=self.test_size, random_state=42
        )
        
        # Second split: separate validation from train
        train_triples, val_triples = train_test_split(
            train_val_triples, test_size=self.val_size/(1-self.test_size), random_state=42
        )
        
        # Save splits
        train_path = os.path.join(self.output_dir, 'train.txt')
        val_path = os.path.join(self.output_dir, 'valid.txt')
        test_path = os.path.join(self.output_dir, 'test.txt')
        
        with open(train_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(train_triples) + '\n')
            
        with open(val_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(val_triples) + '\n')
            
        with open(test_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(test_triples) + '\n')
        
        print(f"Data split completed:")
        print(f"  Train: {len(train_triples)} triples -> {train_path}")
        print(f"  Validation: {len(val_triples)} triples -> {val_path}")
        print(f"  Test: {len(test_triples)} triples -> {test_path}")
        
        return train_path, val_path, test_path

def test_model(model: torch.nn.Module, data_generator: torch_data.DataLoader, 
               entities_count: int, summary_writer: tensorboard.SummaryWriter, 
               device: torch.device, epoch_id: int, metric_suffix: str) -> Tuple[float, float, float, float]:
    """Test model performance with link prediction metrics."""
    examples_count = 0.0
    hits_at_1 = 0.0
    hits_at_3 = 0.0
    hits_at_10 = 0.0
    mrr = 0.0

    model.eval()
    entity_ids = torch.arange(end=entities_count, device=device).unsqueeze(0)
    
    with torch.no_grad():
        for head, relation, tail in data_generator:
            current_batch_size = head.size()[0]

            head, relation, tail = head.to(device), relation.to(device), tail.to(device)
            all_entities = entity_ids.repeat(current_batch_size, 1)
            heads = head.reshape(-1, 1).repeat(1, all_entities.size()[1])
            relations = relation.reshape(-1, 1).repeat(1, all_entities.size()[1])
            tails = tail.reshape(-1, 1).repeat(1, all_entities.size()[1])

            # Check all possible tails
            triplets = torch.stack((heads, relations, all_entities), dim=2).reshape(-1, 3)
            tails_predictions = model.predict(triplets).reshape(current_batch_size, -1)
            
            # Check all possible heads
            triplets = torch.stack((all_entities, relations, tails), dim=2).reshape(-1, 3)
            heads_predictions = model.predict(triplets).reshape(current_batch_size, -1)

            # Concat predictions
            predictions = torch.cat((tails_predictions, heads_predictions), dim=0)
            ground_truth_entity_id = torch.cat((tail.reshape(-1, 1), head.reshape(-1, 1)))

            hits_at_1 += metric.hit_at_k(predictions, ground_truth_entity_id, device=device, k=1)
            hits_at_3 += metric.hit_at_k(predictions, ground_truth_entity_id, device=device, k=3)
            hits_at_10 += metric.hit_at_k(predictions, ground_truth_entity_id, device=device, k=10)
            mrr += metric.mrr(predictions, ground_truth_entity_id)

            examples_count += predictions.size()[0]

    hits_at_1_score = hits_at_1 / examples_count * 100
    hits_at_3_score = hits_at_3 / examples_count * 100
    hits_at_10_score = hits_at_10 / examples_count * 100
    mrr_score = mrr / examples_count * 100
    
    # Log metrics
    summary_writer.add_scalar('Metrics/Hits_1/' + metric_suffix, hits_at_1_score, global_step=epoch_id)
    summary_writer.add_scalar('Metrics/Hits_3/' + metric_suffix, hits_at_3_score, global_step=epoch_id)
    summary_writer.add_scalar('Metrics/Hits_10/' + metric_suffix, hits_at_10_score, global_step=epoch_id)
    summary_writer.add_scalar('Metrics/MRR/' + metric_suffix, mrr_score, global_step=epoch_id)

    return hits_at_1_score, hits_at_3_score, hits_at_10_score, mrr_score

def train_transe(train_path: str, val_path: str, test_path: str, 
                 lr: float = 0.01, batch_size: int = 128, vector_length: int = 100,
                 margin: float = 1.0, norm: int = 1, epochs: int = 1000,
                 use_gpu: bool = True, validation_freq: int = 25, 
                 checkpoint_path: str = "", tensorboard_log_dir: str = "./runs/lcquad_transe"):
    """Train TransE model on LC-QuAD triples."""
    
    # Set random seed
    torch.random.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Setup device
    device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create entity and relation mappings
    print("Creating entity and relation mappings...")
    entity2id, relation2id = data.create_mappings(train_path)
    print(f"Entities: {len(entity2id)}, Relations: {len(relation2id)}")
    
    # Save mappings for later use
    mappings_dir = os.path.dirname(train_path)
    entity_mapping_path = os.path.join(mappings_dir, 'entity2id.json')
    relation_mapping_path = os.path.join(mappings_dir, 'relation2id.json')
    
    with open(entity_mapping_path, 'w', encoding='utf-8') as f:
        json.dump(entity2id, f, ensure_ascii=False, indent=2)
    with open(relation_mapping_path, 'w', encoding='utf-8') as f:
        json.dump(relation2id, f, ensure_ascii=False, indent=2)
    
    print(f"Saved entity mapping to: {entity_mapping_path}")
    print(f"Saved relation mapping to: {relation_mapping_path}")
    
    # Create datasets
    train_set = data.FB15KDataset(train_path, entity2id, relation2id)
    train_generator = torch_data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    
    validation_set = data.FB15KDataset(val_path, entity2id, relation2id)
    validation_generator = torch_data.DataLoader(validation_set, batch_size=batch_size)
    
    test_set = data.FB15KDataset(test_path, entity2id, relation2id)
    test_generator = torch_data.DataLoader(test_set, batch_size=batch_size)
    
    # Initialize model
    model = model_definition.TransE(
        entity_count=len(entity2id), 
        relation_count=len(relation2id), 
        dim=vector_length,
        margin=margin,
        device=device, 
        norm=norm
    )
    model = model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr)
    
    print(f"Model: {model}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Setup tensorboard
    os.makedirs(tensorboard_log_dir, exist_ok=True)
    summary_writer = tensorboard.SummaryWriter(log_dir=tensorboard_log_dir)
    
    start_epoch_id = 1
    step = 0
    best_score = 0.0
    
    # Load checkpoint if provided
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from: {checkpoint_path}")
        start_epoch_id, step, best_score = storage.load_checkpoint(checkpoint_path, model, optimizer)
    
    print("Starting training...")
    
    # Training loop
    for epoch_id in range(start_epoch_id, epochs + 1):
        print(f"Epoch {epoch_id}/{epochs}")
        
        model.train()
        epoch_loss = 0.0
        loss_impacting_samples_count = 0
        samples_count = 0
        
        for batch_idx, (local_heads, local_relations, local_tails) in enumerate(train_generator):
            local_heads = local_heads.to(device)
            local_relations = local_relations.to(device) 
            local_tails = local_tails.to(device)

            positive_triples = torch.stack((local_heads, local_relations, local_tails), dim=1)

            # Generate negative samples by corrupting heads or tails
            head_or_tail = torch.randint(high=2, size=local_heads.size(), device=device)
            random_entities = torch.randint(high=len(entity2id), size=local_heads.size(), device=device)
            broken_heads = torch.where(head_or_tail == 1, random_entities, local_heads)
            broken_tails = torch.where(head_or_tail == 0, random_entities, local_tails)
            negative_triples = torch.stack((broken_heads, local_relations, broken_tails), dim=1)

            optimizer.zero_grad()

            loss, pd, nd = model(positive_triples, negative_triples)
            loss_mean = loss.mean()
            loss_mean.backward()

            # Log metrics
            summary_writer.add_scalar('Loss/train', loss_mean.item(), global_step=step)
            summary_writer.add_scalar('Distance/positive', pd.mean().item(), global_step=step)
            summary_writer.add_scalar('Distance/negative', nd.mean().item(), global_step=step)

            epoch_loss += loss_mean.item()
            loss_cpu = loss.data.cpu()
            loss_impacting_samples_count += loss_cpu.nonzero().size()[0]
            samples_count += loss_cpu.size()[0]

            optimizer.step()
            step += 1
            
            if batch_idx % 100 == 0:
                print(f"  Batch {batch_idx}, Loss: {loss_mean.item():.4f}")

        avg_epoch_loss = epoch_loss / len(train_generator)
        loss_impact_percentage = loss_impacting_samples_count / samples_count * 100
        
        summary_writer.add_scalar('Metrics/loss_impacting_samples', loss_impact_percentage, global_step=epoch_id)
        summary_writer.add_scalar('Loss/epoch_avg', avg_epoch_loss, global_step=epoch_id)
        
        print(f"  Average Loss: {avg_epoch_loss:.4f}")
        print(f"  Loss Impacting Samples: {loss_impact_percentage:.2f}%")

        # Validation
        if epoch_id % validation_freq == 0:
            print("  Running validation...")
            hits_1, hits_3, hits_10, mrr_score = test_model(
                model=model, 
                data_generator=validation_generator,
                entities_count=len(entity2id),
                device=device, 
                summary_writer=summary_writer,
                epoch_id=epoch_id, 
                metric_suffix="val"
            )
            
            print(f"  Validation - Hits@1: {hits_1:.2f}%, Hits@3: {hits_3:.2f}%, Hits@10: {hits_10:.2f}%, MRR: {mrr_score:.2f}%")
            
            # Save best model based on Hits@10
            if hits_10 > best_score:
                best_score = hits_10
                checkpoint_file = "best_lcquad_transe_checkpoint.tar"
                storage.save_checkpoint(model, optimizer, epoch_id, step, best_score)
                print(f"  New best model saved! Hits@10: {best_score:.2f}%")

    # Final testing on best model
    print("\nTesting best model on test set...")
    if os.path.exists("best_lcquad_transe_checkpoint.tar"):
        storage.load_checkpoint("best_lcquad_transe_checkpoint.tar", model, optimizer)
    
    model.eval()
    test_scores = test_model(
        model=model, 
        data_generator=test_generator, 
        entities_count=len(entity2id), 
        device=device,
        summary_writer=summary_writer, 
        epoch_id=epochs, 
        metric_suffix="test"
    )
    
    print(f"Final Test Scores:")
    print(f"  Hits@1: {test_scores[0]:.2f}%")
    print(f"  Hits@3: {test_scores[1]:.2f}%") 
    print(f"  Hits@10: {test_scores[2]:.2f}%")
    print(f"  MRR: {test_scores[3]:.2f}%")
    
    # Save final embeddings
    embeddings_dir = os.path.join(os.path.dirname(train_path), 'embeddings')
    os.makedirs(embeddings_dir, exist_ok=True)
    
    entity_embeddings_path = os.path.join(embeddings_dir, 'entity_embeddings.pt')
    relation_embeddings_path = os.path.join(embeddings_dir, 'relation_embeddings.pt')
    
    torch.save(model.entities_emb.weight.data.cpu(), entity_embeddings_path)
    torch.save(model.relations_emb.weight.data.cpu(), relation_embeddings_path)
    
    print(f"Saved entity embeddings to: {entity_embeddings_path}")
    print(f"Saved relation embeddings to: {relation_embeddings_path}")
    
    summary_writer.close()
    
    return model, entity2id, relation2id, test_scores

def main():
    parser = argparse.ArgumentParser(description='Train TransE on LC-QuAD extracted triples')
    parser.add_argument('--triples_file', default='data/processed/knowledge_graph_triples.txt',
                      help='Path to extracted triples file')
    parser.add_argument('--output_dir', default='data/processed/splits',
                      help='Directory to save train/val/test splits')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--vector_length', type=int, default=100, help='Embedding dimension')
    parser.add_argument('--margin', type=float, default=1.0, help='Margin for ranking loss')
    parser.add_argument('--norm', type=int, default=1, help='Norm for distance calculation (1 or 2)')
    parser.add_argument('--epochs', type=int, default=500, help='Number of training epochs')
    parser.add_argument('--validation_freq', type=int, default=25, help='Validation frequency')
    parser.add_argument('--use_gpu', action='store_true', default=True, help='Use GPU if available')
    parser.add_argument('--checkpoint_path', default='', help='Path to checkpoint to resume from')
    parser.add_argument('--skip_split', action='store_true', help='Skip data splitting if already done')
    
    args = parser.parse_args()
    
    # Phase 1: Data preprocessing and splitting
    if not args.skip_split:
        print("=" * 60)
        print("PHASE B.1: Data Preprocessing and Splitting")
        print("=" * 60)
        
        preprocessor = LCQuADDataPreprocessor(args.triples_file, args.output_dir)
        train_path, val_path, test_path = preprocessor.split_triples()
    else:
        train_path = os.path.join(args.output_dir, 'train.txt')
        val_path = os.path.join(args.output_dir, 'valid.txt')
        test_path = os.path.join(args.output_dir, 'test.txt')
        print(f"Using existing splits:")
        print(f"  Train: {train_path}")
        print(f"  Validation: {val_path}")
        print(f"  Test: {test_path}")
    
    # Phase 2: TransE training
    print("\n" + "=" * 60)
    print("PHASE B.2: TransE Training")
    print("=" * 60)
    
    model, entity2id, relation2id, test_scores = train_transe(
        train_path=train_path,
        val_path=val_path,
        test_path=test_path,
        lr=args.lr,
        batch_size=args.batch_size,
        vector_length=args.vector_length,
        margin=args.margin,
        norm=args.norm,
        epochs=args.epochs,
        use_gpu=args.use_gpu,
        validation_freq=args.validation_freq,
        checkpoint_path=args.checkpoint_path
    )
    
    print("\n" + "=" * 60)
    print("PHASE B: TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print(f"Final model performance:")
    print(f"  Entities: {len(entity2id)}")
    print(f"  Relations: {len(relation2id)}")
    print(f"  Test Hits@1: {test_scores[0]:.2f}%")
    print(f"  Test Hits@3: {test_scores[1]:.2f}%")
    print(f"  Test Hits@10: {test_scores[2]:.2f}%")
    print(f"  Test MRR: {test_scores[3]:.2f}%")

if __name__ == '__main__':
    main() 