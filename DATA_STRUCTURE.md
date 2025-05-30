# GRAPHITE Data Structure Documentation

This document provides a comprehensive overview of the data organization and structure for the GRAPHITE histopathology analysis pipeline.

## 📊 Overview

The GRAPHITE pipeline requires three main datasets organized for different stages of the analysis:

1. **Training Dataset Step 1**: For MIL classification training
2. **Training Dataset Step 2**: For self-supervised learning 
3. **Visualization Dataset**: For explainable AI and attention visualization

## 📁 Directory Structure

```
dataset/
├── training_dataset_step_1/         # MIL Classification Data
│   └── tma_core/                    # Tissue Microarray Core Images
│       ├── 10025/                   # Cancer Patient ID
│       │   ├── patch_001.png        # Individual patches
│       │   ├── patch_002.png
│       │   └── ...
│       ├── 10026/                   # Cancer Patient ID
│       ├── ...                      # More cancer patients
│       ├── 20001/                   # Normal Patient ID
│       │   ├── patch_001.png
│       │   └── ...
│       └── ...                      # More normal patients
│
├── training_dataset_step_2/         # Self-Supervised Learning Data
│   └── core_image/                  # Full core images for SSL training
│       ├── image_001.png
│       ├── image_002.png
│       └── ...
│
├── visualization_dataset/           # Visualization Analysis Data
│   ├── core_image/                  # Images for XAI analysis
│   │   ├── vis_001.png
│   │   ├── vis_002.png
│   │   └── ...
│   └── mask/                        # Ground truth masks for evaluation
│       ├── vis_mask_001.png
│       ├── vis_mask_002.png
│       └── ...
│
├── cancer.txt                       # Cancer patient labels
└── normal.txt                       # Normal patient labels
```

## 🏥 Training Dataset Step 1 (MIL Classification)

### Purpose
This dataset is used for training the Multiple Instance Learning (MIL) model that classifies tissue samples as cancerous or normal based on patch-level analysis.

### Structure
```
training_dataset_step_1/tma_core/
├── Cancer Patients (prefix 10xxx, 22xxx)
│   ├── 10025/
│   ├── 10026/
│   ├── 10027/
│   ├── 10028/
│   ├── 10029/
│   ├── 10030/
│   ├── 10031/
│   ├── 10032/
│   ├── 10033/
│   ├── 10034/
│   ├── 22021/
│   └── 22107/
└── Normal Patients (prefix 20xxx)
    ├── 20001/
    ├── 20002/
    ├── 20003/
    ├── 20004/
    ├── 20005/
    ├── 20006/
    ├── 20007/
    └── 20008/
```

### Data Specifications

#### Image Requirements
- **Format**: PNG, JPG, JPEG, TIFF, TIF
- **Size**: Variable (typically 224x224 to 512x512 pixels)
- **Color**: RGB (3-channel)
- **Patches per Patient**: Typically 50-200 patches
- **Resolution**: High-resolution microscopy images

#### Naming Convention
- **Patient Folders**: Use patient ID as folder name
- **Patch Files**: Sequential naming (e.g., `patch_001.png`, `patch_002.png`)
- **Alternative**: Any descriptive naming is acceptable

#### Label Assignment
- **Cancer Patients**: Patient IDs starting with 10xxx and 22xxx
- **Normal Patients**: Patient IDs starting with 20xxx
- **Label Files**: `cancer.txt` and `normal.txt` contain patient IDs

### Expected Content
Each patient folder should contain:
- Multiple image patches extracted from tissue microarray cores
- Patches should represent diverse regions of the tissue sample
- Quality patches with minimal artifacts and good tissue preservation

## 🧬 Training Dataset Step 2 (Self-Supervised Learning)

### Purpose
This dataset is used for training the hierarchical Graph Attention Network (HierGAT) using self-supervised learning techniques. **No ground truth labels or masks are required** since the model learns representations from the data structure itself.

### Structure
```
training_dataset_step_2/
└── core_image/                      # Full tissue core images
    ├── image_001.png
    ├── image_002.png
    ├── image_003.png
    └── ...
```

### Data Specifications

#### Core Images
- **Format**: PNG, JPG, JPEG, TIFF, TIF
- **Size**: Typically 1024x1024 to 2048x2048 pixels
- **Color**: RGB (3-channel)
- **Content**: Complete tissue microarray core images
- **Quality**: High-resolution, well-focused images

#### Naming Convention
- **Core Images**: `image_XXX.png` where XXX is a sequential number
- **No Masks Required**: Self-supervised learning doesn't need ground truth annotations

### Data Preparation Notes
- Only core images are needed - no masks or labels required
- Remove images with significant artifacts or poor tissue preservation
- Ensure consistent image quality across the dataset
- Images should contain sufficient tissue content for meaningful self-supervised learning

## 👁️ Visualization Dataset

### Purpose
This dataset is used for generating explainable AI visualizations and testing attention fusion algorithms. **Masks are required here for evaluation purposes.**

### Structure
```
visualization_dataset/
├── core_image/                      # Images for XAI analysis
│   ├── vis_001.png
│   ├── vis_002.png
│   └── ...
└── mask/                            # Ground truth masks for evaluation
    ├── vis_mask_001.png
    ├── vis_mask_002.png
    └── ...
```

### Data Specifications

#### Visualization Images
- **Format**: PNG, JPG, JPEG, TIFF, TIF
- **Size**: Consistent with training data (512x512 to 2048x2048)
- **Color**: RGB (3-channel)
- **Content**: Representative tissue samples for analysis
- **Quality**: High-quality images suitable for detailed visualization

#### Ground Truth Masks
- **Format**: PNG (binary masks)
- **Size**: Same dimensions as corresponding images
- **Content**: Annotated regions of interest or tissue boundaries
- **Values**: Binary (0/255) or multi-class annotations
- **Purpose**: Evaluation of attention map quality

## 📝 Label Files

### cancer.txt
```csv
patient_id
10025
10026
10027
10028
10029
10030
10031
10032
10033
10034
22021
22107
```

### normal.txt
```csv
patient_id
20001
20002
20003
20004
20005
20006
20007
20008
```

### File Format
- **Format**: CSV with header
- **Column**: `patient_id`
- **Content**: One patient ID per row
- **Encoding**: UTF-8
- **Location**: Root dataset directory

## 🔧 Data Preparation Guidelines

### 1. Image Quality Standards
- **Resolution**: Minimum 224x224 pixels
- **Focus**: Sharp, well-focused images
- **Staining**: Consistent H&E staining
- **Artifacts**: Minimal bubbles, tears, or debris
- **Exposure**: Proper illumination, no over/under-exposure

### 2. File Organization
- **Consistent Naming**: Use systematic naming conventions
- **No Spaces**: Avoid spaces in file/folder names
- **Case Sensitivity**: Be consistent with capitalization
- **Special Characters**: Avoid special characters in names

### 3. Data Validation
- **File Integrity**: Check all images can be opened
- **Label Consistency**: Ensure patient IDs match folder names (Step 1 only)
- **Missing Files**: Check for any missing or corrupted files
- **Mask Correspondence**: Verify mask-image pairs match (Visualization dataset only)

### 4. Storage Recommendations
- **Backup**: Maintain multiple copies of your data
- **Version Control**: Track data versions and changes
- **Documentation**: Keep metadata about image sources
- **Access Control**: Ensure appropriate data access permissions

## 📊 Dataset Statistics

### Typical Dataset Sizes
- **Step 1 (MIL)**: 
  - Patients: 20 (12 cancer, 8 normal)
  - Total Patches: 1,000-4,000
  - Storage: 5-20 GB

- **Step 2 (SSL)**: 
  - Images: 100-1,000 core images
  - Storage: 10-50 GB
  - **Note**: No masks needed, reducing storage requirements

- **Visualization**: 
  - Images: 50-200 samples
  - Storage: 2-10 GB (including masks)

### Performance Considerations
- **Loading Time**: Optimize image sizes for memory constraints
- **Batch Processing**: Consider batch sizes based on image dimensions
- **Preprocessing**: Plan for data augmentation and normalization
- **Caching**: Implement data caching for repeated access

## 🔄 Data Preprocessing Pipeline

### Step 1: Image Preprocessing
1. **Resize**: Standardize image dimensions
2. **Normalize**: Apply color/intensity normalization
3. **Augment**: Optional data augmentation
4. **Quality Check**: Filter low-quality images

### Step 2: Self-Supervised Learning Preprocessing
1. **Image Only**: No mask processing required
2. **Quality Check**: Filter low-quality images
3. **Resize**: Standardize dimensions if needed
4. **Augment**: Self-supervised augmentations (rotation, color jitter, etc.)

### Step 3: Visualization Preprocessing
1. **Image-Mask Alignment**: Ensure perfect correspondence
2. **Validate**: Check mask coverage and quality
3. **Filter**: Remove invalid mask-image pairs
4. **Document**: Record preprocessing steps

## 🚨 Common Issues and Solutions

### Issue 1: Missing Masks for Step 2
**Solution**: No masks are needed for self-supervised learning - only core images

### Issue 2: Inconsistent Image Sizes
**Solution**: Standardize dimensions during preprocessing

### Issue 3: Missing Patient Data
**Solution**: Update label files to match available data (Step 1 only)

### Issue 4: Poor Image Quality
**Solution**: Implement quality assessment and filtering

### Issue 5: Large File Sizes
**Solution**: Compress images or use efficient formats

## 📋 Data Validation Checklist

Before running the pipeline:

**Step 1 (MIL Classification):**
- [ ] All patient folders exist and contain images
- [ ] Label files contain correct patient IDs
- [ ] File formats are supported
- [ ] Image quality meets standards

**Step 2 (Self-Supervised Learning):**
- [ ] Core images directory exists and contains images
- [ ] No masks required
- [ ] Image quality meets standards
- [ ] Sufficient number of images for training

**Visualization:**
- [ ] Image-mask pairs are properly matched
- [ ] Ground truth masks are accurate
- [ ] File formats are supported

**General:**
- [ ] Directory structure matches expected format
- [ ] No corrupted or empty files
- [ ] Sufficient storage space available
- [ ] Backup copies are available

## 📞 Support

For data structure questions:
- Check the main [README.md](README.md) for setup instructions
- Review the [SETUP.md](SETUP.md) for installation guidance
- Create an issue for specific data organization problems

---

**Note**: Ensure compliance with data privacy regulations and obtain appropriate permissions before using medical imaging data. 