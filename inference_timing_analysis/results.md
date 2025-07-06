**General Settings**  
- **Image Size:** 5040 x 5040  
- **Patch Size:** 224 x 224  
- **Total Patches:** 484  
- **GPU:** V100 (FP32)  
- **Inference:** Per Core  

| Model Name                     | Total Time (s) | FLOPs (GFLOPs)* | Parameters (M) | Model Size (MB) |
|--------------------------------|----------------|------------------|----------------|-----------------|
| GRADCAM                        | 14.00          | 1761.76*         | 11.18           | 42.64           |
| GRADCAM++                      | 7.00           | 1761.76*         | 11.18           | 42.64           |
| SCORECAM                       | 88.00          | 1761.76*         | 11.18           | 42.64           |
| HIRESCAM                       | 7.00           | 1761.76*         | 11.18           | 42.64           |
| ABLATIONCAM                    | 90.00          | 1761.76*         | 11.18           | 42.64           |
| XGRADCAM                       | 7.00           | 1761.76*         | 11.18           | 42.64           |
| EIGENCAM                       | 7.00           | 1761.76*         | 11.18           | 42.64           |
| FULLGRAD                       | 359.00         | 1761.76*         | 11.18           | 42.64           |
| SHAP_DEEP                      | 48.00          | 1761.76*         | 11.18           | 42.64           |
| SHAP_GRADIENT                  | 572.75         | 1761.76*         | 11.18           | 42.64           |
| LIME                           | 318.64         | 1761.76*         | 11.18           | 42.64           |
| MIL_ATTENTION                  | 0.52           | 1766.80*         | 12.03           | 45.91           |
| GRAPHITE (Multilevel Fusion)   | 0.72           | 1766.92*         | 11.41           | 43.53           |
| GRAPHITE (Final Fusion)        | 360.38         | 5295.48*         | 12.20           | 46.54           |

\* **Note:** FLOPs represent the base model’s computational cost and do not reflect the complete computational cost.
