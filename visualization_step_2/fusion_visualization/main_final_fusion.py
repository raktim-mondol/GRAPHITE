import random
import numpy as np
import torch
import os
def set_seed(seed=25):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)

set_seed()


# For DataLoader initialization:
#generator = torch.Generator().manual_seed(78)

from utils.imports import *

import torch
import os
from pathlib import Path
from models.hiergat import HierGATSSL
from data.slide_processor import CustomSlide
from utils.model_utils import load_mil_model
from data.dataset import CoreImageProcessor
from models.graph_builder import HierarchicalGraphBuilder
import logging
import json
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from data.patch_extractor import MultiscalePatchExtractor
from models.graph_builder import HierarchicalGraphBuilder
from models.hiergat import HierGATSSL
from training.trainer import HierGATSSLTrainer
from models.hiergat import HierGATSSL
from utils.visualization import *
import argparse
from models.graph_builder import HierarchicalGraphBuilder
from data.dataset import CoreImageProcessor, HierGATSSLDataset
from training.losses import HierarchicalInfoMaxLoss, LossTracker
from torch_geometric.loader import DataLoader as PyGDataLoader
from models.inference import HierGATSSLInference
from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from utils.extract_core_mask import CoreExtractor
import pandas as pd
from openpyxl.styles import Font, PatternFill, Alignment


def main():
    """
    Main function for visualization with consistent model loading.
    
    This implementation follows the step_1_part_2 pattern for consistency
    in MIL and CAM visualization methods.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Generate multi-attention visualizations')
    parser.add_argument('--cam_method', type=str, default='fullgrad',
                       choices=['fullgrad', 'gradcam', 'hirescam', 'scorecam', 'gradcampp', 'gradcam++', 
                               'ablationcam', 'xgradcam', 'eigencam'],
                       help='CAM method to use for visualization (default: fullgrad)')
    parser.add_argument('--fusion_method', type=str, default='confidence',
                       choices=['optimal', 'weighted', 'adaptive', 'multiscale', 'confidence'],
                       help='Fusion method for final attention heatmap (default: confidence)')
    parser.add_argument('--model_path', type=str, 
                       default="../../training_step_2/self_supervised_training/output/best_model.pt",
                       help='Path to the GAT model')
    parser.add_argument('--mil_model_path', type=str,
                       default="../../training_step_1/mil_classification/output/best_model.pth", 
                       help='Path to the MIL model')
    parser.add_argument('--dataset_dir', type=str,
                       default="../../dataset/training_dataset_step_1/tma_core",
                       help='Directory containing images to process')
    parser.add_argument('--save_dir', type=str,
                       default="./output/visualization_results",
                       help='Directory to save visualizations')
    parser.add_argument('--mask_dir', type=str,
                       default="../../dataset/training_dataset_step_2/mask",
                       help='Directory containing annotation masks')
    parser.add_argument('--calculate_metrics', action='store_true', default=True,
                       help='Calculate performance metrics for fusion methods (default: True)')
    parser.add_argument('--metrics_thresholds', type=float, nargs='+', 
                       default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                       help='Thresholds for metrics calculation (default: 0.1 to 0.9 in 0.1 steps)')
    
    args = parser.parse_args()
    
    # Set seeds for reproducibility
    model_path = args.model_path
    mil_model_path = args.mil_model_path
    dataset_dir = args.dataset_dir
    save_dir = args.save_dir
    mask_dir = args.mask_dir
    cam_method = args.cam_method
    fusion_method = args.fusion_method
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Using CAM method: {cam_method.upper()}")
    print(f"Using fusion method: {fusion_method.upper()}")
    print(f"Calculate metrics: {args.calculate_metrics}")
    if args.calculate_metrics:
        print(f"Metrics thresholds: {args.metrics_thresholds}")
    
    # Initialize models using consistent loading approach
    gat_inference = HierGATSSLInference(model_path, mil_model_path=mil_model_path)
    
    try:
        print(f"Loading MIL model from: {mil_model_path}")
        # Use consistent model loading with proper parameters
        mil_model = load_mil_model(
            model_path=mil_model_path,
            device=device,
            num_classes=2,
            feat_dim=512,  # ResNet18 feature dimension
            proj_dim=128
        )
        print(f"✅ MIL model loaded successfully")
        
    except Exception as e:
        print(f"❌ Error loading MIL model: {str(e)}")
        raise
    
    # Process images
    results_summary = []
    successful_with_metrics = []
    
    for image_path in Path(dataset_dir).glob('*.png'):
        try:
            print(f"\nProcessing {image_path.name} with {cam_method.upper()}")
            
            # Get GAT results
            gat_results = gat_inference.process_image(str(image_path))
            
            # Combine attention maps with consistent MIL and CAM processing
            result = combine_all_attention_maps(
                gat_results, 
                mil_model, 
                str(image_path), 
                device,
                save_dir,
                mask_dir=mask_dir,
                cam_method=cam_method,
                fusion_method=fusion_method,
                calculate_metrics=args.calculate_metrics,
                metrics_thresholds=args.metrics_thresholds
            )
            
            # Unpack results (handle new return format)
            if result and len(result) == 4:
                final_fusion_heatmap, smooth_maps, final_fusion_weights, fusion_metrics = result
            else:
                final_fusion_heatmap = None
                fusion_metrics = None
            
            if final_fusion_heatmap is not None:
                print(f"Successfully processed {image_path.name}")
                print("Final fusion weights:", final_fusion_weights)
                print(f"Final fusion heatmap shape: {final_fusion_heatmap.shape}")
                print(f"Attention range: [{final_fusion_heatmap.min():.4f}, {final_fusion_heatmap.max():.4f}]")
                
                # Store result for summary
                result_entry = {
                    'image': image_path.name,
                    'status': 'success',
                    'fusion_weights': final_fusion_weights,
                    'attention_stats': {
                        'min': float(final_fusion_heatmap.min()),
                        'max': float(final_fusion_heatmap.max()),
                        'mean': float(final_fusion_heatmap.mean()),
                        'std': float(final_fusion_heatmap.std())
                    }
                }
                
                # Display metrics summary if available
                if fusion_metrics:
                    print(f"📊 Performance Metrics Summary:")
                    metrics_summary = {}
                    for method_name, method_data in fusion_metrics.items():
                        # Get best metrics across all thresholds
                        best_f1 = max([m['f1_score'] for m in method_data.values()])
                        best_iou = max([m['iou'] for m in method_data.values()])
                        best_precision = max([m['precision'] for m in method_data.values()])
                        best_recall = max([m['recall'] for m in method_data.values()])
                        
                        display_name = method_name.replace('_', ' ')
                        print(f"   📈 {display_name}: F1={best_f1:.3f}, IoU={best_iou:.3f}, P={best_precision:.3f}, R={best_recall:.3f}")
                        
                        metrics_summary[method_name] = {
                            'best_f1': best_f1,
                            'best_iou': best_iou,
                            'best_precision': best_precision,
                            'best_recall': best_recall
                        }
                    
                    result_entry['metrics_summary'] = metrics_summary
                    successful_with_metrics.append(result_entry)
                
                results_summary.append(result_entry)
                
            else:
                print(f"⚠️ Failed to process {image_path.name}")
                results_summary.append({
                    'image': image_path.name,
                    'status': 'failed'
                })
                
        except Exception as e:
            print(f"❌ Error processing {image_path.name}: {str(e)}")
            import traceback
            traceback.print_exc()
            results_summary.append({
                'image': image_path.name,
                'status': 'error',
                'error': str(e)
            })
            continue

    # Create comprehensive summary
    total_images = len(list(Path(dataset_dir).glob('*.png')))
    successful_images = len([r for r in results_summary if r['status'] == 'success'])
    failed_images = total_images - successful_images
    
    print(f"\n🎉 Visualization processing completed with {cam_method.upper()} and {fusion_method.upper()} fusion!")
    print(f"📊 Processing Summary:")
    print(f"   Total images: {total_images}")
    print(f"   Successful: {successful_images}")
    print(f"   Failed: {failed_images}")
    print(f"   With metrics: {len(successful_with_metrics)}")
    
    # Save results summary
    if save_dir:
        import json
        summary_path = Path(save_dir) / f'processing_summary_{cam_method}_{fusion_method}.json'
        
        # Calculate overall metrics statistics if available
        overall_metrics = {}
        if successful_with_metrics:
            for method_name in ['Multilevel_Fusion', 'MIL_Attention', 'CAM_Based', 'Final_Fusion_Heatmap']:
                method_metrics = [r['metrics_summary'][method_name] for r in successful_with_metrics if 'metrics_summary' in r and method_name in r['metrics_summary']]
                
                if method_metrics:
                    overall_metrics[method_name] = {
                        'avg_f1': np.mean([m['best_f1'] for m in method_metrics]),
                        'std_f1': np.std([m['best_f1'] for m in method_metrics]),
                        'avg_iou': np.mean([m['best_iou'] for m in method_metrics]),
                        'std_iou': np.std([m['best_iou'] for m in method_metrics]),
                        'avg_precision': np.mean([m['best_precision'] for m in method_metrics]),
                        'std_precision': np.std([m['best_precision'] for m in method_metrics]),
                        'avg_recall': np.mean([m['best_recall'] for m in method_metrics]),
                        'std_recall': np.std([m['best_recall'] for m in method_metrics]),
                        'num_samples': len(method_metrics)
                    }
        
        summary_data = {
            'cam_method': cam_method,
            'fusion_method': fusion_method,
            'calculate_metrics': args.calculate_metrics,
            'metrics_thresholds': args.metrics_thresholds if args.calculate_metrics else None,
            'total_images': total_images,
            'successful': successful_images,
            'failed': failed_images,
            'with_metrics': len(successful_with_metrics),
            'overall_metrics': overall_metrics,
            'results': results_summary
        }
        
        with open(summary_path, 'w') as f:
            json.dump(summary_data, f, indent=2)
        print(f"\n📊 Results summary saved to: {summary_path}")
        
        # Create overall metrics Excel file if we have metrics
        if overall_metrics:
            overall_excel_path = Path(save_dir) / f'overall_fusion_metrics_summary_{cam_method}_{fusion_method}.xlsx'
            with pd.ExcelWriter(overall_excel_path, engine='openpyxl') as writer:
                
                # Create summary DataFrame
                summary_data_list = []
                for method_name, metrics in overall_metrics.items():
                    summary_data_list.append({
                        'Fusion_Method': method_name.replace('_', ' '),
                        'Avg_F1': round(metrics['avg_f1'], 4),
                        'Std_F1': round(metrics['std_f1'], 4),
                        'Avg_IoU': round(metrics['avg_iou'], 4),
                        'Std_IoU': round(metrics['std_iou'], 4),
                        'Avg_Precision': round(metrics['avg_precision'], 4),
                        'Std_Precision': round(metrics['std_precision'], 4),
                        'Avg_Recall': round(metrics['avg_recall'], 4),
                        'Std_Recall': round(metrics['std_recall'], 4),
                        'Num_Samples': metrics['num_samples']
                    })
                
                df_summary = pd.DataFrame(summary_data_list)
                df_summary.to_excel(writer, sheet_name='Overall_Summary', index=False)
                
                # Format the sheet
                workbook = writer.book
                worksheet = writer.sheets['Overall_Summary']
                
                # Format headers
                header_fill = PatternFill(start_color='366092', end_color='366092', fill_type='solid')
                header_font = Font(color='FFFFFF', bold=True)
                
                for cell in worksheet[1]:
                    cell.fill = header_fill
                    cell.font = header_font
                    cell.alignment = Alignment(horizontal='center')
                
                # Auto-adjust column widths
                for column in worksheet.columns:
                    max_length = 0
                    column_letter = column[0].column_letter
                    for cell in column:
                        try:
                            if len(str(cell.value)) > max_length:
                                max_length = len(str(cell.value))
                        except:
                            pass
                    adjusted_width = min(max_length + 2, 15)
                    worksheet.column_dimensions[column_letter].width = adjusted_width
            
            print(f"📊 Overall fusion metrics Excel saved to: {overall_excel_path}")
            
            # Print overall metrics summary
            print(f"\n📈 Overall Performance Summary (CAM: {cam_method.upper()}):")
            for method_name, metrics in overall_metrics.items():
                print(f"   {method_name.replace('_', ' ')}:")
                print(f"      F1: {metrics['avg_f1']:.3f} ± {metrics['std_f1']:.3f}")
                print(f"      IoU: {metrics['avg_iou']:.3f} ± {metrics['std_iou']:.3f}")
                print(f"      Precision: {metrics['avg_precision']:.3f} ± {metrics['std_precision']:.3f}")
                print(f"      Recall: {metrics['avg_recall']:.3f} ± {metrics['std_recall']:.3f}")
                print(f"      Samples: {metrics['num_samples']}")


if __name__ == "__main__":
    main()