from utils.imports import *

        
# In training/losses.py

class HierarchicalInfoMaxLoss(nn.Module):
    def __init__(self, 
                 temperature=0.07,
                 alpha=0.5,
                 beta=0.5,
                 scale_temp=0.1,
                 eps=1e-8):  # Add small epsilon to prevent numerical instability
        super().__init__()
        self.temperature = temperature
        self.scale_temp = scale_temp
        self.alpha = alpha
        self.beta = beta
        self.eps = eps
        
    def _compute_scale_loss(self, node_embeddings, level_indices, scale_weights):
        """Compute Scale-wise loss components with safeguards"""
        device = node_embeddings.device
        scale_losses = []
        
        # Inter-scale consistency
        for i in range(len(level_indices)):
            for j in range(i+1, len(level_indices)):
                indices_i = level_indices[i]
                indices_j = level_indices[j]
                
                if len(indices_i) < 2 or len(indices_j) < 2:
                    continue
                    
                emb_i = F.normalize(node_embeddings[indices_i], dim=1)
                emb_j = F.normalize(node_embeddings[indices_j], dim=1)
                
                sim_matrix = torch.mm(emb_i, emb_j.T) / self.scale_temp
                
                # Add numerical stability
                sim_matrix = torch.clamp(sim_matrix, min=-20, max=20)
                
                # Weight by scale attention
                weight_factor = scale_weights[i] * scale_weights[j]
                
                level_loss = -torch.log(
                    torch.exp(sim_matrix).mean() + self.eps
                ) * weight_factor
                
                if not torch.isnan(level_loss).any():  # Check if any element is NaN
                    scale_losses.append(level_loss)
        
        # Intra-scale discrimination
        for i, indices in enumerate(level_indices):
            if len(indices) > 2:
                emb = F.normalize(node_embeddings[indices], dim=1)
                sim_matrix = torch.mm(emb, emb.T) / self.scale_temp
                
                mask = torch.eye(len(indices), device=device)
                sim_matrix = sim_matrix * (1 - mask)
                
                # Add numerical stability
                sim_matrix = torch.clamp(sim_matrix, min=-20, max=20)
                
                level_loss = -torch.log(
                    1 - torch.exp(sim_matrix).mean() + self.eps
                ) * scale_weights[i]
                
                if not torch.isnan(level_loss).any():  # Check if any element is NaN
                    scale_losses.append(level_loss)
        
        if len(scale_losses) > 0:
            return torch.stack(scale_losses).mean()
        else:
            return torch.tensor(0.0, device=device)
    
    def _compute_infomax_loss(self, node_embeddings, graph_embedding, level_indices):
        """Compute InfoMax loss components with safeguards"""
        device = node_embeddings.device
        infomax_losses = []
        
        for level_idx, indices in enumerate(level_indices):
            if len(indices) < 2:
                continue
                
            level_nodes = node_embeddings[indices]
            level_graph = graph_embedding[level_idx:level_idx+1]
            
            level_nodes = F.normalize(level_nodes, dim=1)
            level_graph = F.normalize(level_graph, dim=1)
            
            similarity = torch.sum(level_nodes * level_graph, dim=1)
            similarity = torch.clamp(similarity, min=-20, max=20)
            pos_score = torch.exp(similarity / self.temperature)
            
            neg_similarity = torch.mm(level_nodes, level_graph.T)
            neg_similarity = torch.clamp(neg_similarity, min=-20, max=20)
            neg_score = torch.exp(neg_similarity / self.temperature).sum(dim=1)
            
            level_loss = -torch.log((pos_score + self.eps) / (pos_score + neg_score + self.eps)).mean()
            
            if not torch.isnan(level_loss).any():
                infomax_losses.append(level_loss)
            
            if len(indices) > 2:
                sim_matrix = torch.mm(level_nodes, level_nodes.T)
                sim_matrix = torch.clamp(sim_matrix, min=-20, max=20)
                sim_matrix = sim_matrix / self.temperature
                
                mask = torch.eye(len(indices), device=device)
                sim_matrix = sim_matrix * (1 - mask)
                
                pos_sim = sim_matrix.diagonal()
                neg_sim = torch.exp(sim_matrix).sum(dim=1) - torch.exp(pos_sim)
                
                local_local_loss = -torch.log(
                    (torch.exp(pos_sim) + self.eps) / 
                    (torch.exp(pos_sim) + neg_sim + self.eps)
                ).mean()
                
                if not torch.isnan(local_local_loss).any():
                    infomax_losses.append(local_local_loss)
        
        if len(infomax_losses) > 0:
            return torch.stack(infomax_losses).mean()
        else:
            return torch.tensor(0.0, device=device)
    
    def forward(self, outputs):
        """Compute combined hierarchical loss with safeguards"""
        node_emb = outputs['node_embeddings']
        graph_emb = outputs['graph_embedding']
        
        level_indices = []
        current_idx = 0
        for i in range(3):
            level_attn = outputs['attention_weights'][f'level_{i}']
            num_nodes = level_attn.size(0)
            level_indices.append(torch.arange(current_idx, current_idx + num_nodes, device=node_emb.device))
            current_idx += num_nodes
            
        infomax_loss = self._compute_infomax_loss(node_emb, graph_emb, level_indices)
        scale_loss = self._compute_scale_loss(
            node_emb,
            level_indices,
            outputs['attention_weights']['cross_scale']
        )
        
        # Combine losses with safety checks
        if torch.isnan(scale_loss).any():
            print("Warning: Scale loss is NaN, using only InfoMax loss")
            total_loss = infomax_loss
        else:
            total_loss = self.alpha * infomax_loss + self.beta * scale_loss
        
        return {
            'total_loss': total_loss,
            'infomax_loss': infomax_loss,
            'scale_loss': scale_loss
        }
        
    
              
        
        
# Loss tracker for visualization
class LossTracker:
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.losses = {
            'total_loss': [],
            'infomax_loss': [],
            'scale_loss': []
        }
        
    def update(self, loss_dict):
        for key, value in loss_dict.items():
            self.losses[key].append(value.item())
    
    def get_average_losses(self):
        return {
            key: sum(values) / len(values) if values else 0 
            for key, values in self.losses.items()
        }
        
    def plot_losses(self, save_path):
        plt.figure(figsize=(10, 5))
        for key, values in self.losses.items():
            plt.plot(values, label=key)
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(save_path)
        plt.close()