"""
Dataset and DataLoader utilities for MIL-based histopathology image classification.
"""

import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset, Sampler
from PIL import Image
from sklearn.model_selection import train_test_split
import logging
from torch.utils.data._utils.collate import default_collate

logger = logging.getLogger(__name__)

class NormalizationError(Exception):
    """Exception raised when image normalization fails."""
    pass

class PatientDataset(Dataset):
    """
    Dataset class for loading patient patch images.
    Each patient is represented as a bag of image patches.
    
    Attributes:
        root_dir (str): Root directory containing patient folders
        patient_ids (list): List of patient identifiers
        labels (DataFrame): DataFrame with patient_id as index and 'label' column
        transform (callable, optional): Optional transform to apply to the image
        model_transform (callable, optional): Transform specific to the model requirements
        color_normalization (callable, optional): Color normalization transform
        max_patches (int, optional): Maximum number of patches to use per patient
        tissue_threshold (float): Minimum tissue percentage required for a patch
    """
    
    def __init__(self, root_dir, patient_ids, labels, transform=None, model_transform=None, 
                 color_normalization=None, max_patches=None, tissue_threshold=0.5):
        """
        Initialize the PatientDataset.
        
        Args:
            root_dir (str): Root directory containing patient folders
            patient_ids (list): List of patient identifiers
            labels (DataFrame): DataFrame with patient_id as index and 'label' column
            transform (callable, optional): Optional transform to apply to the image
            model_transform (callable, optional): Transform specific to the model
            color_normalization (callable, optional): Color normalization transform
            max_patches (int, optional): Maximum number of patches to use per patient
            tissue_threshold (float): Minimum tissue percentage for patch inclusion
        """
        self.root_dir = root_dir
        self.patient_ids = patient_ids
        self.labels = labels
        self.transform = transform
        self.model_transform = model_transform
        self.color_normalization = color_normalization
        self.max_patches = max_patches
        self.tissue_threshold = tissue_threshold
        
        # Set up patient patch data
        self.patients_patches = {}
        self.patients_labels = {}
        total_patches = 0
        kept_patches = 0
        
        for patient_id in self.patient_ids:
            # Get all patches for this patient
            patient_dir = os.path.join(self.root_dir, str(patient_id))
            if not os.path.isdir(patient_dir):
                logger.warning(f"Directory not found for patient {patient_id}: {patient_dir}")
                continue
                
            patch_files = glob.glob(os.path.join(patient_dir, "*.png")) + \
                         glob.glob(os.path.join(patient_dir, "*.jpg"))
            
            if not patch_files:
                logger.warning(f"No image files found for patient {patient_id} in {patient_dir}")
                continue
                
            total_patches += len(patch_files)
              # Skip tissue filtering for faster loading - just randomly sample patches
            if self.max_patches and len(patch_files) > self.max_patches:
                # Randomly sample patches without tissue filtering for speed
                filtered_patches = np.random.choice(
                    patch_files, self.max_patches, replace=False
                ).tolist()
            else:
                filtered_patches = patch_files
            
            kept_patches += len(filtered_patches)
            
            if filtered_patches:  # Only add if there are valid patches
                self.patients_patches[patient_id] = filtered_patches
                self.patients_labels[patient_id] = self.labels.loc[patient_id, 'label']
        
        # Update patient_ids to only include those with valid patches
        self.patient_ids = list(self.patients_patches.keys())
        
        logger.info(f"Loaded {kept_patches}/{total_patches} patches for {len(self.patient_ids)} patients")
        logger.info(f"Average patches per patient: {kept_patches/max(1,len(self.patient_ids)):.1f}")
    
    def _has_sufficient_tissue(self, img, threshold=None):
        """
        Check if an image has sufficient tissue content.
        
        Args:
            img (PIL.Image): The image to check
            threshold (float, optional): Tissue percentage threshold, defaults to self.tissue_threshold
            
        Returns:
            bool: True if the image has sufficient tissue, False otherwise
        """
        if threshold is None:
            threshold = self.tissue_threshold
            
        # Convert to numpy array and to grayscale if needed
        img_array = np.array(img)
        if len(img_array.shape) == 3 and img_array.shape[2] == 3:
            # Simple grayscale conversion: avg of RGB
            gray = np.mean(img_array, axis=2)
        else:
            gray = img_array
            
        # Simple tissue detection: non-white pixels
        # White is typically 255, so we threshold to detect non-background
        tissue_mask = gray < 220
        tissue_percentage = np.mean(tissue_mask)
        
        return tissue_percentage >= threshold
    
    def __len__(self):
        """Return the number of patients in the dataset."""
        return len(self.patient_ids)
    
    def __getitem__(self, idx):
        """
        Get a patient's patches and label.
        
        Args:
            idx (int): Index of the patient
            
        Returns:
            tuple: (patches, label, patient_id)
                patches (torch.Tensor): Tensor of shape [num_patches, channels, height, width]                label (int): Class label for the patient
                patient_id (str): ID of the patient
        """
        patient_id = self.patient_ids[idx]
        patch_files = self.patients_patches[patient_id]
        label = self.patients_labels[patient_id]
        patches = []
        for patch_file in patch_files:
            try:
                img = Image.open(patch_file).convert('RGB')
                
                # Quick tissue check - skip completely white patches
                img_array = np.array(img)
                if np.mean(img_array) > 240:  # Skip mostly white patches
                    continue
                
                # Apply color normalization if specified
                if self.color_normalization:
                    try:
                        img = self.color_normalization(img)
                    except NormalizationError:
                        # Skip this patch if normalization fails
                        continue
                  # Apply general transforms
                if self.transform:
                    img = self.transform(img)
                # Apply model-specific transforms
                if self.model_transform:
                    img = self.model_transform(img)
                patches.append(img)
            except Exception as e:
                logger.warning(f"Error processing image {patch_file}: {e}")
        
        if not patches:
            # If no valid patches, create a dummy tensor
            # This is a fallback to prevent errors, but logging is needed
            logger.warning(f"No valid patches for patient {patient_id}, using zeros")
            if self.model_transform:
                # Try to infer tensor shape from the model transform
                dummy = torch.zeros((3, 224, 224))  # Assume default size
            else:
                dummy = torch.zeros((3, 224, 224))
            patches = [dummy]
        
        # Convert any PIL images to tensors
        processed_patches = []
        for patch in patches:
            if isinstance(patch, torch.Tensor):
                processed_patches.append(patch)
            elif isinstance(patch, Image.Image):
                # Convert PIL image to tensor using ToTensor transform
                from torchvision import transforms
                to_tensor = transforms.ToTensor()
                patch_tensor = to_tensor(patch)
                processed_patches.append(patch_tensor)
            else:
                logger.warning(f"Skipping patch of unsupported type: {type(patch)}")
        
        if not processed_patches:
            logger.warning(f"No valid patches after conversion for patient {patient_id}, using zeros")
            processed_patches = [torch.zeros((3, 224, 224))]
            
        # Stack all patches for this patient
        patches_tensor = torch.stack(processed_patches)
        
        return patches_tensor, label, patient_id


class BalancedBatchSampler(Sampler):
    """
    Sampler to ensure balanced class representation in each batch.
    For each batch, it samples an equal number of patients from each class.
    
    Attributes:
        dataset (Dataset): Dataset to sample from
        batch_size (int): Size of batches to generate
        drop_last (bool): Whether to drop the last incomplete batch
    """
    
    def __init__(self, dataset, batch_size, drop_last=False):
        """
        Initialize the balanced batch sampler.
        
        Args:
            dataset (Dataset): Dataset to sample from
            batch_size (int): Size of batches to generate
            drop_last (bool): Whether to drop the last incomplete batch
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        
        # Group indices by label
        self.label_to_indices = {}
        for idx, (_, label, _) in enumerate(self.dataset):
            if label not in self.label_to_indices:
                self.label_to_indices[label] = []
            self.label_to_indices[label].append(idx)
        
        # Number of unique labels
        self.num_classes = len(self.label_to_indices)
        
        # Ensure batch_size is divisible by num_classes
        if self.batch_size % self.num_classes != 0:
            logger.warning(
                f"Batch size {self.batch_size} is not divisible by number of classes {self.num_classes}. "
                f"Some batches may not be perfectly balanced."
            )
        
        # Number of samples per class per batch
        self.samples_per_class = self.batch_size // self.num_classes
        
        # Minimum number of samples for a label
        self.min_label_samples = min(len(indices) for indices in self.label_to_indices.values())
        
        # Calculate the number of batches
        if drop_last:
            self.num_batches = self.min_label_samples // self.samples_per_class
        else:
            self.num_batches = (self.min_label_samples + self.samples_per_class - 1) // self.samples_per_class
    
    def __iter__(self):
        """
        Yield balanced batches of indices.
        
        Returns:
            iterator: Iterator over batches of indices
        """
        # Shuffle indices for each label
        label_indices = {
            label: np.random.permutation(indices).tolist()
            for label, indices in self.label_to_indices.items()
        }
        
        # Current position in each label's indices
        position = {label: 0 for label in label_indices}
        
        for _ in range(self.num_batches):
            batch_indices = []
            
            # Sample from each class
            for label in label_indices:
                samples_to_take = min(
                    self.samples_per_class,
                    len(label_indices[label]) - position[label]
                )
                
                if samples_to_take > 0:
                    batch_indices.extend(
                        label_indices[label][position[label]:position[label] + samples_to_take]
                    )
                    position[label] += samples_to_take
                
                # If we've used all samples for this label, reshuffle
                if position[label] >= len(label_indices[label]):
                    label_indices[label] = np.random.permutation(
                        self.label_to_indices[label]
                    ).tolist()
                    position[label] = 0
            
            # Check if batch is too small and needs to be dropped
            if len(batch_indices) < self.batch_size and self.drop_last:
                continue
                
            yield batch_indices
    
    def __len__(self):
        """Return the number of batches."""
        return self.num_batches


def custom_collate(batch):
    """
    Custom collate function for PatientDataset.
    Handles batches where each patient has a different number of patches.
    
    Args:
        batch: List of (patches, label, patient_id) tuples
        
    Returns:
        tuple: (patches_list, labels, patient_ids)
            patches_list: List of tensors, each of shape [num_patches, channels, height, width]
            labels: Tensor of shape [batch_size]
            patient_ids: List of patient IDs
    """
    # Separate the batch into individual components
    patches = [item[0] for item in batch]  # List of tensors with shape [num_patches, C, H, W]
    labels = torch.tensor([item[1] for item in batch])  # Convert labels to tensor
    patient_ids = [item[2] for item in batch]
    
    # Return without stacking the patches (each element in patches is already a tensor)
    # The model needs to handle variable-sized patch tensors
    return patches, labels, patient_ids


def setup_dataloaders(root_dir, patient_ids, labels, batch_size=8, max_patches=500,
                     test_size=0.3, random_state=78, use_balanced_sampler=False,
                     color_normalization=None, model_transform=None):
    """
    Set up data loaders for training and testing with the specified train-test split.
    
    Args:
        root_dir (str): Root directory containing patient folders
        patient_ids (list): List of patient identifiers
        labels (DataFrame): DataFrame with patient_id as index and 'label' column
        batch_size (int): Batch size
        max_patches (int, optional): Maximum number of patches per patient
        test_size (float): Proportion of the dataset to use for testing (default: 0.3)
        random_state (int): Random seed for reproducibility (default: 78)
        use_balanced_sampler (bool): Whether to use the BalancedBatchSampler
        color_normalization (callable, optional): Color normalization transform
        model_transform (callable, optional): Transform specific to the model
        
    Returns:
        tuple: (train_loader, test_loader) - DataLoaders for training and testing
    """
    # Create the full dataset first
    dataset = PatientDataset(
        root_dir=root_dir,
        patient_ids=patient_ids,
        labels=labels,
        model_transform=model_transform,
        color_normalization=color_normalization,
        max_patches=max_patches
    )
    
    # Create labels list for stratification
    dataset_labels = [labels.loc[patient_id, 'label'] for patient_id in dataset.patient_ids]
    
    # Create train-test split using indices
    train_indices, test_indices = train_test_split(
        range(len(dataset)), 
        test_size=test_size, 
        stratify=dataset_labels, 
        random_state=random_state
    )
    
    # Create subsets
    train_subset = Subset(dataset, train_indices)
    test_subset = Subset(dataset, test_indices)
    
    logger.info(f"Split dataset: {len(train_indices)} train patients, {len(test_indices)} test patients")
    
    # Create data loaders
    if use_balanced_sampler:
        # For balanced sampler, we need to create a new sampler for the subset
        # This is more complex and may not work directly with Subset
        logger.warning("Balanced sampler with Subset may not work as expected. Using regular DataLoader.")
        train_loader = DataLoader(
            train_subset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
            collate_fn=custom_collate
        )
    else:
        train_loader = DataLoader(
            train_subset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
            collate_fn=custom_collate
        )
    
    test_loader = DataLoader(
        test_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        collate_fn=custom_collate
    )
    
    return train_loader, test_loader
