from utils.imports import *

        
# In training/losses.py

class HierarchicalInfoMaxLoss(nn.Module):
    def __init__(self, 
                 temperature=0.07,
                 alpha=0.5,
                 beta=0.5,
                 tau=0.1,  # Temperature parameter τ for scalewise loss
                 eps=1e-8):  # Add small epsilon to prevent numerical instability
        super().__init__()
        self.temperature = temperature
        self.tau = tau  # Temperature parameter τ from the equation
        self.alpha = alpha
        self.beta = beta
        self.eps = eps
        
    def _compute_scale_loss(self, node_embeddings, level_indices, scale_weights):
        """
        Compute Scale-wise loss according to the updated equation:
        L_scale = -∑_{m=0}^{M-1} ∑_{l=m+1}^{M-1} w_m w_l × log(exp(s_{ii}^{ml}/τ) / (∑_{j≠i} exp(s_{ij}^{ml}/τ) + ε))
        
        This maintains consistency across magnification levels through:
        (1) interscale consistency (similarity between embeddings of the same node at adjacent scales)
        (2) intrascale discrimination (dissimilarity between different nodes at the same scale)
        """
        device = node_embeddings.device
        total_loss = torch.tensor(0.0, device=device)
        num_pairs = 0
        
        M = len(level_indices)  # Number of scales
        
        # Iterate over all scale pairs (m, l) where m < l
        for m in range(M):
            for l in range(m + 1, M):
                indices_m = level_indices[m]
                indices_l = level_indices[l]
                
                if len(indices_m) == 0 or len(indices_l) == 0:
                    continue
                
                # Get embeddings for scales m and l
                emb_m = F.normalize(node_embeddings[indices_m], dim=1)  # [N_m, D]
                emb_l = F.normalize(node_embeddings[indices_l], dim=1)  # [N_l, D]
                
                # Compute similarity matrix s_{ij}^{ml} between all nodes at scales m and l
                sim_matrix = torch.mm(emb_m, emb_l.T) / self.tau  # [N_m, N_l]
                
                # Add numerical stability
                sim_matrix = torch.clamp(sim_matrix, min=-20, max=20)
                
                # Get level attention weights
                w_m = scale_weights[m] if m < len(scale_weights) else 1.0
                w_l = scale_weights[l] if l < len(scale_weights) else 1.0
                weight_factor = w_m * w_l
                
                # For each node i, compute the loss term
                min_nodes = min(len(indices_m), len(indices_l))
                node_losses = []
                
                for i in range(min_nodes):
                    # s_{ii}^{ml}: similarity of same node across scales m and l
                    s_ii = sim_matrix[i, i]
                    
                    # Numerator: exp(s_{ii}^{ml}/τ)
                    numerator = torch.exp(s_ii)
                    
                    # Denominator: ∑_{j≠i} exp(s_{ij}^{ml}/τ) + ε
                    # Create mask to exclude diagonal element (j ≠ i)
                    mask = torch.ones_like(sim_matrix[i])
                    mask[i] = 0
                    
                    # Sum over all j ≠ i
                    denominator = torch.sum(torch.exp(sim_matrix[i]) * mask) + self.eps
                    
                    # Compute log term for this node
                    log_term = torch.log(numerator / denominator)
                    node_losses.append(log_term)
                
                if node_losses:
                    # Average over nodes and apply weight factor
                    pair_loss = -weight_factor * torch.stack(node_losses).mean()
                    
                    # Ensure pair_loss is a scalar
                    if pair_loss.dim() > 0:
                        pair_loss = pair_loss.mean()
                    
                    # Check for numerical stability
                    if not torch.isnan(pair_loss).any() and not torch.isinf(pair_loss).any():
                        total_loss = total_loss + pair_loss
                        num_pairs += 1
        
        # Return average loss across all scale pairs
        if num_pairs > 0:
            return total_loss / num_pairs
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