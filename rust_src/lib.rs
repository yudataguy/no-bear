//! Performance-critical components for drone flight core.
//!
//! This crate provides Rust implementations of compute-intensive
//! operations that benefit from native performance.

use pyo3::prelude::*;

mod geometry;
mod image_processing;

/// Calculate haversine distance between two GPS coordinates.
///
/// Returns distance in meters.
#[pyfunction]
fn haversine_distance(lat1: f64, lon1: f64, lat2: f64, lon2: f64) -> f64 {
    geometry::haversine_distance(lat1, lon1, lat2, lon2)
}

/// Calculate bearing between two GPS coordinates.
///
/// Returns bearing in degrees (0-360).
#[pyfunction]
fn calculate_bearing(lat1: f64, lon1: f64, lat2: f64, lon2: f64) -> f64 {
    geometry::calculate_bearing(lat1, lon1, lat2, lon2)
}

/// Calculate destination point given start, bearing, and distance.
///
/// Returns (latitude, longitude) tuple.
#[pyfunction]
fn destination_point(lat: f64, lon: f64, bearing_deg: f64, distance_m: f64) -> (f64, f64) {
    geometry::destination_point(lat, lon, bearing_deg, distance_m)
}

/// Check if a point is inside a polygon (geofence).
///
/// Uses ray casting algorithm for point-in-polygon test.
#[pyfunction]
fn point_in_polygon(lat: f64, lon: f64, polygon: Vec<(f64, f64)>) -> bool {
    geometry::point_in_polygon(lat, lon, &polygon)
}

/// Calculate Intersection over Union (IoU) between two bounding boxes.
///
/// Boxes are in (x, y, width, height) format.
#[pyfunction]
fn calculate_iou(box1: (f64, f64, f64, f64), box2: (f64, f64, f64, f64)) -> f64 {
    image_processing::calculate_iou(box1, box2)
}

/// Non-maximum suppression for bounding boxes.
///
/// Takes list of (x, y, w, h, confidence) and returns indices to keep.
#[pyfunction]
fn nms(boxes: Vec<(f64, f64, f64, f64, f64)>, iou_threshold: f64) -> Vec<usize> {
    image_processing::nms(&boxes, iou_threshold)
}

/// Python module definition.
#[pymodule]
fn _rust_core(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(haversine_distance, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_bearing, m)?)?;
    m.add_function(wrap_pyfunction!(destination_point, m)?)?;
    m.add_function(wrap_pyfunction!(point_in_polygon, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_iou, m)?)?;
    m.add_function(wrap_pyfunction!(nms, m)?)?;
    Ok(())
}
