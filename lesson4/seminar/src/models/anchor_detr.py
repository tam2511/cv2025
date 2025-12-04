import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50
from scipy.optimize import linear_sum_assignment
import math


class PositionEmbedding2D(nn.Module):
    def __init__(self, num_pos_feats=128, temperature=10000):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature

    def forward(self, x):
        B, C, H, W = x.shape
        y_embed = torch.arange(H, dtype=torch.float32, device=x.device).unsqueeze(1).repeat(1, W)
        x_embed = torch.arange(W, dtype=torch.float32, device=x.device).unsqueeze(0).repeat(H, 1)
        
        y_embed = y_embed / H
        x_embed = x_embed / W
        
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)
        
        pos_x = x_embed[:, :, None] / dim_t
        pos_y = y_embed[:, :, None] / dim_t
        
        pos_x = torch.stack([pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()], dim=3).flatten(2)
        pos_y = torch.stack([pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()], dim=3).flatten(2)
        
        pos = torch.cat([pos_y, pos_x], dim=2).permute(2, 0, 1)
        return pos.unsqueeze(0).repeat(B, 1, 1, 1)


def generate_anchor_points(H, W, device):
    anchor_points = []
    for i in range(H):
        for j in range(W):
            cx = (j + 0.5) / W
            cy = (i + 0.5) / H
            anchor_points.append([cx, cy])
    
    anchor_points = torch.tensor(anchor_points, device=device, dtype=torch.float32)
    return anchor_points


def pos2posemb2d(pos, num_pos_feats=128, temperature=10000):
    """
    Sinusoidal position encoding for 2D positions (cx, cy).
    
    Args:
        pos: [..., 2] - normalized (cx, cy) coordinates
        num_pos_feats: dimension for each coordinate
        temperature: temperature for sinusoidal encoding
    
    Returns:
        [..., num_pos_feats * 2] - positional embeddings
    """
    scale = 2 * math.pi
    pos = pos * scale
    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=pos.device)
    dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
    pos_x = pos[..., 0, None] / dim_t
    pos_y = pos[..., 1, None] / dim_t
    pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2)
    pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1).flatten(-2)
    posemb = torch.cat((pos_y, pos_x), dim=-1)
    return posemb


class AnchorEncoderLayer(nn.Module):
    """
    Custom encoder layer for Anchor DETR.
    Positional embeddings are added to query/key but not to value.
    """
    def __init__(self, d_model, nhead, dim_feedforward, dropout, batch_first=True):
        super().__init__()
        
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first)
        
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        self.activation = nn.ReLU()
    
    def forward(self, src, pos):
        """
        Args:
            src: source features [B, HW, C]
            pos: positional embeddings [B, HW, C]
        """
        # Self-attention with positional embeddings
        # Add pos to query and key, but not to value
        q = k = src + pos
        src2 = self.self_attn(q, k, src)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        
        # FFN
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        
        return src


class AnchorDecoderLayer(nn.Module):
    """
    Custom decoder layer for Anchor DETR.
    Positional embeddings are computed from reference_points and added inside the layer.
    """
    def __init__(self, d_model, nhead, dim_feedforward, dropout, batch_first=True):
        super().__init__()
        
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first)
        
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        
        self.activation = nn.ReLU()
    
    def forward(self, tgt, memory, memory_pos, query_pos):
        """
        Args:
            tgt: content queries [B, N, C]
            memory: encoder output [B, HW, C] WITHOUT positional encoding
            memory_pos: positional embeddings for memory [B, HW, C]
            query_pos: positional embeddings for queries [B, N, C]
        """
        # Self-attention with positional embeddings
        q = k = tgt + query_pos
        tgt2 = self.self_attn(q, k, tgt)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        
        # Cross-attention
        # Add pos to memory key, but not to value (like in encoder)
        tgt2 = self.multihead_attn(tgt + query_pos, memory + memory_pos, memory)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        
        # FFN
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        
        return tgt


class ResNetFPN(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        backbone = resnet50(pretrained=pretrained)
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        
        self.conv_out = nn.Conv2d(2048, 256, kernel_size=1)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.conv_out(x)
        return x


def box_cxcywh_to_xyxy(boxes):
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack([cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h], dim=-1)


def generalized_box_iou(boxes1, boxes2):
    boxes1 = box_cxcywh_to_xyxy(boxes1)
    boxes2 = box_cxcywh_to_xyxy(boxes2)
    
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    
    union = area1[:, None] + area2 - inter
    iou = inter / (union + 1e-6)
    
    lt_enclosing = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb_enclosing = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])
    wh_enclosing = (rb_enclosing - lt_enclosing).clamp(min=0)
    area_enclosing = wh_enclosing[:, :, 0] * wh_enclosing[:, :, 1]
    
    giou = iou - (area_enclosing - union) / (area_enclosing + 1e-6)
    return giou


def giou_loss(pred_boxes, target_boxes):
    giou = generalized_box_iou(pred_boxes, target_boxes)
    return (1 - torch.diag(giou)).sum()


class AnchorDETR(nn.Module):
    def __init__(
        self,
        num_classes=80,
        emb_dim=256,
        nhead=8,
        enc_layers=6,
        dec_layers=6,
        num_patterns=3,
        pretrained=True,
        cost_class=2.0,
        cost_bbox=5.0,
        cost_giou=2.0,
        w_class=2.0,
        w_bbox=5.0,
        w_giou=2.0
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.emb_dim = emb_dim
        
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.w_class = w_class
        self.w_bbox = w_bbox
        self.w_giou = w_giou
        
        self.backbone = ResNetFPN(pretrained=pretrained)
        
        self.register_buffer('anchor_points', None)
        
        self.pos_emb = PositionEmbedding2D(num_pos_feats=emb_dim // 2)
        
        self.encoder_layers = nn.ModuleList([
            AnchorEncoderLayer(emb_dim, nhead, emb_dim * 4, dropout=0.1, batch_first=True)
            for _ in range(enc_layers)
        ])
        
        self.decoder_layers = nn.ModuleList([
            AnchorDecoderLayer(emb_dim, nhead, emb_dim * 4, dropout=0.1, batch_first=True)
            for _ in range(dec_layers)
        ])
        self.num_decoder_layers = dec_layers
        
        self.pattern = nn.Embedding(num_patterns, emb_dim)
        
        self.adapt_pos2d = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim),
        )
        
        self.class_head = nn.Linear(emb_dim, num_classes + 1)
        
        # Single bbox head for all 4 coordinates (cx, cy, w, h)
        self.bbox_head = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, 4)
        )
        
        self._reset_parameters()
    
    def _reset_parameters(self):
        """Initialize weights following the original Anchor DETR paper."""
        # Initialize class head bias to favor background
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        nn.init.constant_(self.class_head.bias.data[-1], bias_value)
        
        # Initialize bbox head
        nn.init.constant_(self.bbox_head[-1].weight.data, 0)
        nn.init.constant_(self.bbox_head[-1].bias.data, 0)
        # Bias for w, h to predict small boxes initially
        nn.init.constant_(self.bbox_head[-1].bias.data[2:], -2.0)
    
    def forward(self, x):
        features = self.backbone(x)
        B, C, H, W = features.shape
        
        if self.anchor_points is None or self.anchor_points.shape[0] != H * W:
            self.anchor_points = generate_anchor_points(H, W, x.device)
        
        pos_emb = self.pos_emb(features)
        
        mem = features.flatten(2).permute(0, 2, 1)
        mem_pos = pos_emb.flatten(2).permute(0, 2, 1)
        
        # Encoder: pass src and pos separately
        # Positional info is added to query/key but not to value
        mem_encoded = mem
        for layer in self.encoder_layers:
            mem_encoded = layer(mem_encoded, mem_pos)
        
        num_patterns = self.pattern.weight.shape[0]
        tgt = self.pattern.weight.reshape(1, num_patterns, 1, self.emb_dim).repeat(B, 1, H * W, 1).reshape(B, num_patterns * H * W, self.emb_dim)
        
        anchor_points_expanded = self.anchor_points.unsqueeze(1).repeat(1, num_patterns, 1).reshape(H * W * num_patterns, 2)
        reference_points = anchor_points_expanded.unsqueeze(0).expand(B, -1, -1)
        
        output = tgt
        if not hasattr(self, '_fwd_call_count'):
            self._fwd_call_count = 0
        self._fwd_call_count += 1
        
        log_this_call = (self._fwd_call_count % 100 == 1)
        
        if log_this_call:
            print(f"\n=== Forward Call {self._fwd_call_count} ===")
            print(f"tgt: mean={tgt.mean().item():.4f}, std={tgt.std().item():.4f}")
            print(f"mem_encoded: mean={mem_encoded.mean().item():.4f}, std={mem_encoded.std().item():.4f}")
            print(f"reference_points: min={reference_points.min().item():.4f}, max={reference_points.max().item():.4f}")
        
        for lid, layer in enumerate(self.decoder_layers):
            query_pos = pos2posemb2d(reference_points, num_pos_feats=self.emb_dim // 2)
            query_pos = self.adapt_pos2d(query_pos)
            output = layer(output, mem_encoded, mem_pos, query_pos)
            
            if log_this_call:
                print(f"Layer {lid}: output mean={output.mean().item():.4f}, std={output.std().item():.4f}")
                print(f"Layer {lid}: query_pos mean={query_pos.mean().item():.4f}, std={query_pos.std().item():.4f}")
        
        decoder_output = output
        
        class_logits = self.class_head(decoder_output)
        
        # Bbox prediction: predict all 4 coords, add inverse_sigmoid(anchor) to center
        bbox_raw = self.bbox_head(decoder_output)
        
        eps = 1e-5
        ref_clamped = reference_points.clamp(eps, 1 - eps)
        inverse_sigmoid_ref = torch.logit(ref_clamped)
        
        # Add inverse_sigmoid anchor to center prediction
        bbox_raw[..., :2] = bbox_raw[..., :2] + inverse_sigmoid_ref
        
        # Apply sigmoid to get final bbox
        bbox_pred = torch.sigmoid(bbox_raw)
        
        if log_this_call:
            print(f"\nOutput Statistics:")
            print(f"class_logits: mean={class_logits.mean().item():.4f}, std={class_logits.std().item():.4f}")
            probs = F.softmax(class_logits, dim=-1)
            fg_prob = 1 - probs[..., -1]
            print(f"FG prob: mean={fg_prob.mean().item():.4f}, max={fg_prob.max().item():.4f}")
            print(f"bbox_raw (before sigmoid): mean={bbox_raw.mean().item():.4f}, std={bbox_raw.std().item():.4f}")
            print(f"bbox_pred: cx=[{bbox_pred[..., 0].min().item():.3f}, {bbox_pred[..., 0].max().item():.3f}], "
                  f"cy=[{bbox_pred[..., 1].min().item():.3f}, {bbox_pred[..., 1].max().item():.3f}], "
                  f"w=[{bbox_pred[..., 2].min().item():.3f}, {bbox_pred[..., 2].max().item():.3f}], "
                  f"h=[{bbox_pred[..., 3].min().item():.3f}, {bbox_pred[..., 3].max().item():.3f}]")
        
        return {
            'class_logits': class_logits,
            'bbox_pred': bbox_pred,
            'anchor_points': reference_points
        }
    
    def match(self, outputs, targets):
        B = outputs['class_logits'].shape[0]
        log_prob = F.log_softmax(outputs['class_logits'], dim=-1)
        bbox_pred = outputs['bbox_pred']
        
        if not hasattr(self, '_match_call_count'):
            self._match_call_count = 0
        self._match_call_count += 1
        log_match = (self._match_call_count % 100 == 1)
        
        indices = []
        for b in range(B):
            if len(targets[b]['labels']) == 0:
                indices.append((torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long)))
                continue
            
            gt_labels = targets[b]['labels']
            gt_boxes = targets[b]['boxes']
            
            cost_class = -log_prob[b][:, gt_labels]
            cost_bbox = torch.cdist(bbox_pred[b], gt_boxes, p=1)
            cost_giou = -generalized_box_iou(bbox_pred[b], gt_boxes)
            
            cost_matrix = self.cost_class * cost_class + self.cost_bbox * cost_bbox + self.cost_giou * cost_giou
            
            if log_match and b == 0:
                print(f"\n=== Matching Call {self._match_call_count} (batch {b}) ===")
                print(f"GT: {len(gt_labels)} objects, labels={gt_labels.tolist()}")
                print(f"GT boxes: {gt_boxes.tolist()}")
                print(f"cost_class: mean={cost_class.mean().item():.4f}, min={cost_class.min().item():.4f}")
                print(f"cost_bbox: mean={cost_bbox.mean().item():.4f}, min={cost_bbox.min().item():.4f}")
                print(f"cost_giou: mean={cost_giou.mean().item():.4f}, min={cost_giou.min().item():.4f}")
                print(f"cost_matrix: mean={cost_matrix.mean().item():.4f}, min={cost_matrix.min().item():.4f}")
            
            pred_idx, gt_idx = linear_sum_assignment(cost_matrix.detach().cpu().numpy())
            
            if log_match and b == 0:
                print(f"Matched {len(pred_idx)} predictions to GT")
                if len(pred_idx) > 0:
                    matched_boxes = bbox_pred[b][pred_idx]
                    print(f"Matched pred boxes: {matched_boxes.tolist()}")
            
            indices.append((torch.tensor(pred_idx, dtype=torch.long), torch.tensor(gt_idx, dtype=torch.long)))
        
        return indices
    
    def compute_loss(self, outputs, targets):
        indices = self.match(outputs, targets)
        
        if not hasattr(self, '_loss_call_count'):
            self._loss_call_count = 0
        self._loss_call_count += 1
        log_loss = (self._loss_call_count % 100 == 1)
        
        B, Q = outputs['class_logits'].shape[:2]
        target_classes = torch.full((B, Q), self.num_classes, dtype=torch.long, device=outputs['class_logits'].device)
        
        for b, (pred_idx, gt_idx) in enumerate(indices):
            if len(pred_idx) > 0:
                target_classes[b, pred_idx] = targets[b]['labels'][gt_idx]
        
        empty_weight = torch.ones(self.num_classes + 1, device=outputs['class_logits'].device)
        empty_weight[-1] = 0.1
        loss_class = F.cross_entropy(outputs['class_logits'].flatten(0, 1), target_classes.flatten(), weight=empty_weight)
        
        loss_bbox, loss_giou, num_boxes = 0, 0, 0
        for b, (pred_idx, gt_idx) in enumerate(indices):
            if len(pred_idx) == 0:
                continue
            pred_boxes = outputs['bbox_pred'][b, pred_idx]
            gt_boxes = targets[b]['boxes'][gt_idx]
            
            l1_loss = F.l1_loss(pred_boxes, gt_boxes, reduction='sum')
            giou_loss_val = giou_loss(pred_boxes, gt_boxes)
            
            loss_bbox += l1_loss
            loss_giou += giou_loss_val
            num_boxes += len(pred_idx)
        
        if num_boxes > 0:
            loss_bbox /= num_boxes
            loss_giou /= num_boxes
        
        total_loss = self.w_class * loss_class + self.w_bbox * loss_bbox + self.w_giou * loss_giou
        
        if log_loss:
            print(f"\n=== Loss Call {self._loss_call_count} ===")
            print(f"num_boxes (total matched): {num_boxes}")
            print(f"loss_class: {loss_class.item():.4f}")
            if num_boxes > 0:
                print(f"loss_bbox: {loss_bbox.item():.4f}")
                print(f"loss_giou: {loss_giou.item():.4f}")
            else:
                print(f"loss_bbox: 0.0 (no matches)")
                print(f"loss_giou: 0.0 (no matches)")
            print(f"total_loss: {total_loss.item():.4f}")
        
        return total_loss
    
    @torch.no_grad()
    def postprocess(self, outputs, conf_threshold=0.5):
        class_logits = outputs['class_logits']
        bbox_pred = outputs['bbox_pred']
        
        probs = F.softmax(class_logits, dim=-1)
        scores, labels = probs[:, :, :-1].max(dim=-1)
        
        results = []
        for b in range(class_logits.shape[0]):
            keep = scores[b] >= conf_threshold
            results.append({
                'boxes': bbox_pred[b][keep],
                'scores': scores[b][keep],
                'labels': labels[b][keep]
            })
        
        return results

