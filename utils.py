import torch

def pack_pathway_output(frames):
    """
    Splits video tensors safely into Slow and Fast pathways without channel corruption.
    Expected Input: 4D [C, T, H, W] or 5D [B, C, T, H, W]
    """
    alpha = 4
    # Dynamically track the temporal axis configuration
    temporal_axis = 2 if frames.dim() == 5 else 1
    
    fast_pathway = frames
    num_frames = frames.shape[temporal_axis]
    
    # Generate frame selections cleanly on the device
    indices = torch.linspace(0, num_frames - 1, num_frames // alpha).long().to(frames.device)
    slow_pathway = torch.index_select(frames, temporal_axis, indices)
    
    return [slow_pathway, fast_pathway]