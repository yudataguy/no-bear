//! Image processing utilities for object detection.

/// Calculate Intersection over Union (IoU) between two bounding boxes.
///
/// Boxes are in (x, y, width, height) format.
pub fn calculate_iou(box1: (f64, f64, f64, f64), box2: (f64, f64, f64, f64)) -> f64 {
    let (x1, y1, w1, h1) = box1;
    let (x2, y2, w2, h2) = box2;

    // Convert to corner format
    let box1_x2 = x1 + w1;
    let box1_y2 = y1 + h1;
    let box2_x2 = x2 + w2;
    let box2_y2 = y2 + h2;

    // Calculate intersection
    let inter_x1 = x1.max(x2);
    let inter_y1 = y1.max(y2);
    let inter_x2 = box1_x2.min(box2_x2);
    let inter_y2 = box1_y2.min(box2_y2);

    let inter_width = (inter_x2 - inter_x1).max(0.0);
    let inter_height = (inter_y2 - inter_y1).max(0.0);
    let intersection = inter_width * inter_height;

    // Calculate union
    let area1 = w1 * h1;
    let area2 = w2 * h2;
    let union = area1 + area2 - intersection;

    if union <= 0.0 {
        0.0
    } else {
        intersection / union
    }
}

/// Non-maximum suppression for bounding boxes.
///
/// Takes boxes as (x, y, w, h, confidence) and returns indices to keep.
pub fn nms(boxes: &[(f64, f64, f64, f64, f64)], iou_threshold: f64) -> Vec<usize> {
    if boxes.is_empty() {
        return vec![];
    }

    // Sort by confidence (descending)
    let mut indices: Vec<usize> = (0..boxes.len()).collect();
    indices.sort_by(|&a, &b| {
        boxes[b].4.partial_cmp(&boxes[a].4).unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut keep = Vec::new();
    let mut suppressed = vec![false; boxes.len()];

    for &idx in &indices {
        if suppressed[idx] {
            continue;
        }

        keep.push(idx);

        let box1 = (boxes[idx].0, boxes[idx].1, boxes[idx].2, boxes[idx].3);

        // Suppress overlapping boxes
        for &other_idx in &indices {
            if suppressed[other_idx] || other_idx == idx {
                continue;
            }

            let box2 = (
                boxes[other_idx].0,
                boxes[other_idx].1,
                boxes[other_idx].2,
                boxes[other_idx].3,
            );

            if calculate_iou(box1, box2) > iou_threshold {
                suppressed[other_idx] = true;
            }
        }
    }

    keep
}

/// Apply simple threshold to grayscale image.
///
/// Returns binary mask.
pub fn threshold(image: &[u8], width: usize, height: usize, threshold: u8) -> Vec<u8> {
    image
        .iter()
        .map(|&pixel| if pixel > threshold { 255 } else { 0 })
        .collect()
}

/// Calculate histogram of grayscale image.
pub fn histogram(image: &[u8]) -> [u32; 256] {
    let mut hist = [0u32; 256];
    for &pixel in image {
        hist[pixel as usize] += 1;
    }
    hist
}

/// Calculate Otsu's threshold for image binarization.
pub fn otsu_threshold(image: &[u8]) -> u8 {
    let hist = histogram(image);
    let total = image.len() as f64;

    let mut sum = 0.0;
    for i in 0..256 {
        sum += i as f64 * hist[i] as f64;
    }

    let mut sum_b = 0.0;
    let mut w_b = 0.0;
    let mut max_variance = 0.0;
    let mut threshold = 0u8;

    for i in 0..256 {
        w_b += hist[i] as f64;
        if w_b == 0.0 {
            continue;
        }

        let w_f = total - w_b;
        if w_f == 0.0 {
            break;
        }

        sum_b += i as f64 * hist[i] as f64;

        let m_b = sum_b / w_b;
        let m_f = (sum - sum_b) / w_f;

        let variance = w_b * w_f * (m_b - m_f).powi(2);

        if variance > max_variance {
            max_variance = variance;
            threshold = i as u8;
        }
    }

    threshold
}

/// Simple box blur on grayscale image.
pub fn box_blur(image: &[u8], width: usize, height: usize, radius: usize) -> Vec<u8> {
    let mut output = vec![0u8; width * height];
    let kernel_size = (2 * radius + 1) * (2 * radius + 1);

    for y in 0..height {
        for x in 0..width {
            let mut sum = 0u32;
            let mut count = 0u32;

            for dy in 0..=(2 * radius) {
                for dx in 0..=(2 * radius) {
                    let ny = (y + dy).saturating_sub(radius);
                    let nx = (x + dx).saturating_sub(radius);

                    if ny < height && nx < width {
                        sum += image[ny * width + nx] as u32;
                        count += 1;
                    }
                }
            }

            output[y * width + x] = (sum / count) as u8;
        }
    }

    output
}

/// Find connected components in binary image.
///
/// Returns labeled image and number of components.
pub fn connected_components(image: &[u8], width: usize, height: usize) -> (Vec<u32>, u32) {
    let mut labels = vec![0u32; width * height];
    let mut current_label = 0u32;
    let mut equivalences: Vec<u32> = vec![0]; // Union-find parent array

    // First pass
    for y in 0..height {
        for x in 0..width {
            let idx = y * width + x;

            if image[idx] == 0 {
                continue;
            }

            let mut neighbors = Vec::new();

            // Check left neighbor
            if x > 0 && labels[idx - 1] > 0 {
                neighbors.push(labels[idx - 1]);
            }

            // Check top neighbor
            if y > 0 && labels[idx - width] > 0 {
                neighbors.push(labels[idx - width]);
            }

            if neighbors.is_empty() {
                // New label
                current_label += 1;
                labels[idx] = current_label;
                equivalences.push(current_label);
            } else {
                // Use minimum neighbor label
                let min_label = *neighbors.iter().min().unwrap();
                labels[idx] = min_label;

                // Record equivalences
                for &neighbor in &neighbors {
                    union(&mut equivalences, min_label as usize, neighbor as usize);
                }
            }
        }
    }

    // Find roots for all labels
    for i in 1..=current_label as usize {
        find(&mut equivalences, i);
    }

    // Second pass - resolve labels
    let mut label_map = vec![0u32; current_label as usize + 1];
    let mut final_label = 0u32;

    for i in 1..=current_label as usize {
        let root = find(&mut equivalences, i);
        if label_map[root] == 0 {
            final_label += 1;
            label_map[root] = final_label;
        }
        label_map[i] = label_map[root];
    }

    for label in &mut labels {
        if *label > 0 {
            *label = label_map[*label as usize];
        }
    }

    (labels, final_label)
}

fn find(parent: &mut [u32], i: usize) -> usize {
    if parent[i] as usize != i {
        parent[i] = find(parent, parent[i] as usize) as u32;
    }
    parent[i] as usize
}

fn union(parent: &mut [u32], i: usize, j: usize) {
    let root_i = find(parent, i);
    let root_j = find(parent, j);
    if root_i != root_j {
        parent[root_j] = root_i as u32;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_iou_identical() {
        let iou = calculate_iou((0.0, 0.0, 10.0, 10.0), (0.0, 0.0, 10.0, 10.0));
        assert!((iou - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_iou_no_overlap() {
        let iou = calculate_iou((0.0, 0.0, 10.0, 10.0), (20.0, 20.0, 10.0, 10.0));
        assert!(iou.abs() < 0.001);
    }

    #[test]
    fn test_iou_partial() {
        let iou = calculate_iou((0.0, 0.0, 10.0, 10.0), (5.0, 5.0, 10.0, 10.0));
        // Intersection: 5*5 = 25, Union: 100 + 100 - 25 = 175
        let expected = 25.0 / 175.0;
        assert!((iou - expected).abs() < 0.001);
    }

    #[test]
    fn test_nms_single() {
        let boxes = vec![(0.0, 0.0, 10.0, 10.0, 0.9)];
        let keep = nms(&boxes, 0.5);
        assert_eq!(keep, vec![0]);
    }

    #[test]
    fn test_nms_suppress() {
        let boxes = vec![
            (0.0, 0.0, 10.0, 10.0, 0.9),
            (1.0, 1.0, 10.0, 10.0, 0.8), // Overlapping, should be suppressed
            (50.0, 50.0, 10.0, 10.0, 0.7), // Not overlapping, should be kept
        ];
        let keep = nms(&boxes, 0.5);
        assert!(keep.contains(&0));
        assert!(keep.contains(&2));
        assert!(!keep.contains(&1));
    }

    #[test]
    fn test_threshold() {
        let image = vec![100, 150, 200, 50];
        let result = threshold(&image, 2, 2, 128);
        assert_eq!(result, vec![0, 255, 255, 0]);
    }
}
