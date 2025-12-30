//! Geographic and geometric calculations.

use std::f64::consts::PI;

/// Earth's radius in meters.
const EARTH_RADIUS_M: f64 = 6_371_000.0;

/// Convert degrees to radians.
#[inline]
fn to_radians(degrees: f64) -> f64 {
    degrees * PI / 180.0
}

/// Convert radians to degrees.
#[inline]
fn to_degrees(radians: f64) -> f64 {
    radians * 180.0 / PI
}

/// Calculate haversine distance between two GPS coordinates.
///
/// # Arguments
/// * `lat1`, `lon1` - First coordinate in degrees
/// * `lat2`, `lon2` - Second coordinate in degrees
///
/// # Returns
/// Distance in meters
pub fn haversine_distance(lat1: f64, lon1: f64, lat2: f64, lon2: f64) -> f64 {
    let lat1_rad = to_radians(lat1);
    let lat2_rad = to_radians(lat2);
    let delta_lat = to_radians(lat2 - lat1);
    let delta_lon = to_radians(lon2 - lon1);

    let a = (delta_lat / 2.0).sin().powi(2)
        + lat1_rad.cos() * lat2_rad.cos() * (delta_lon / 2.0).sin().powi(2);
    let c = 2.0 * a.sqrt().atan2((1.0 - a).sqrt());

    EARTH_RADIUS_M * c
}

/// Calculate bearing between two GPS coordinates.
///
/// # Arguments
/// * `lat1`, `lon1` - Start coordinate in degrees
/// * `lat2`, `lon2` - End coordinate in degrees
///
/// # Returns
/// Bearing in degrees (0-360)
pub fn calculate_bearing(lat1: f64, lon1: f64, lat2: f64, lon2: f64) -> f64 {
    let lat1_rad = to_radians(lat1);
    let lat2_rad = to_radians(lat2);
    let delta_lon = to_radians(lon2 - lon1);

    let y = delta_lon.sin() * lat2_rad.cos();
    let x = lat1_rad.cos() * lat2_rad.sin() - lat1_rad.sin() * lat2_rad.cos() * delta_lon.cos();

    let bearing = to_degrees(y.atan2(x));
    (bearing + 360.0) % 360.0
}

/// Calculate destination point given start, bearing, and distance.
///
/// # Arguments
/// * `lat`, `lon` - Start coordinate in degrees
/// * `bearing_deg` - Bearing in degrees
/// * `distance_m` - Distance in meters
///
/// # Returns
/// (latitude, longitude) tuple in degrees
pub fn destination_point(lat: f64, lon: f64, bearing_deg: f64, distance_m: f64) -> (f64, f64) {
    let lat_rad = to_radians(lat);
    let lon_rad = to_radians(lon);
    let bearing_rad = to_radians(bearing_deg);
    let angular_distance = distance_m / EARTH_RADIUS_M;

    let dest_lat = (lat_rad.sin() * angular_distance.cos()
        + lat_rad.cos() * angular_distance.sin() * bearing_rad.cos())
    .asin();

    let dest_lon = lon_rad
        + (bearing_rad.sin() * angular_distance.sin() * lat_rad.cos())
            .atan2(angular_distance.cos() - lat_rad.sin() * dest_lat.sin());

    (to_degrees(dest_lat), to_degrees(dest_lon))
}

/// Check if a point is inside a polygon using ray casting.
///
/// # Arguments
/// * `lat`, `lon` - Point to test
/// * `polygon` - List of (lat, lon) vertices forming the polygon
///
/// # Returns
/// true if point is inside polygon
pub fn point_in_polygon(lat: f64, lon: f64, polygon: &[(f64, f64)]) -> bool {
    if polygon.len() < 3 {
        return false;
    }

    let mut inside = false;
    let mut j = polygon.len() - 1;

    for i in 0..polygon.len() {
        let (yi, xi) = polygon[i];
        let (yj, xj) = polygon[j];

        if ((yi > lon) != (yj > lon)) && (lat < (xj - xi) * (lon - yi) / (yj - yi) + xi) {
            inside = !inside;
        }
        j = i;
    }

    inside
}

/// Calculate distance from a point to a line segment.
///
/// # Arguments
/// * `px`, `py` - Point coordinates
/// * `x1`, `y1` - Line segment start
/// * `x2`, `y2` - Line segment end
///
/// # Returns
/// Minimum distance to line segment
pub fn point_to_line_distance(px: f64, py: f64, x1: f64, y1: f64, x2: f64, y2: f64) -> f64 {
    let dx = x2 - x1;
    let dy = y2 - y1;
    let length_sq = dx * dx + dy * dy;

    if length_sq == 0.0 {
        // Line segment is a point
        return ((px - x1).powi(2) + (py - y1).powi(2)).sqrt();
    }

    // Project point onto line
    let t = ((px - x1) * dx + (py - y1) * dy) / length_sq;
    let t = t.clamp(0.0, 1.0);

    let proj_x = x1 + t * dx;
    let proj_y = y1 + t * dy;

    ((px - proj_x).powi(2) + (py - proj_y).powi(2)).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_haversine_known_distance() {
        // New York to Los Angeles (approximately 3935 km)
        let ny_lat = 40.7128;
        let ny_lon = -74.0060;
        let la_lat = 34.0522;
        let la_lon = -118.2437;

        let distance = haversine_distance(ny_lat, ny_lon, la_lat, la_lon);
        assert!((distance - 3_935_000.0).abs() < 50_000.0); // Within 50km
    }

    #[test]
    fn test_haversine_same_point() {
        let distance = haversine_distance(0.0, 0.0, 0.0, 0.0);
        assert!(distance < 0.001);
    }

    #[test]
    fn test_bearing_north() {
        let bearing = calculate_bearing(0.0, 0.0, 1.0, 0.0);
        assert!((bearing - 0.0).abs() < 0.1);
    }

    #[test]
    fn test_bearing_east() {
        let bearing = calculate_bearing(0.0, 0.0, 0.0, 1.0);
        assert!((bearing - 90.0).abs() < 0.1);
    }

    #[test]
    fn test_point_in_polygon() {
        let polygon = vec![
            (0.0, 0.0),
            (0.0, 10.0),
            (10.0, 10.0),
            (10.0, 0.0),
        ];

        assert!(point_in_polygon(5.0, 5.0, &polygon));
        assert!(!point_in_polygon(15.0, 15.0, &polygon));
    }
}
