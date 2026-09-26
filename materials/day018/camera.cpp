#include "camera.hpp"

#include <cmath>
#include <stdexcept>

namespace day018 {

Camera::Camera(double fx_px, double fy_px, double cx_px, double cy_px)
    : fx_px_{fx_px}, fy_px_{fy_px}, cx_px_{cx_px}, cy_px_{cy_px} {
    if (!std::isfinite(fx_px_) || !std::isfinite(fy_px_) ||
        !std::isfinite(cx_px_) || !std::isfinite(cy_px_)) {
        throw std::invalid_argument{"camera parameters must be finite"};
    }
    if (fx_px_ <= 0.0 || fy_px_ <= 0.0) {
        throw std::invalid_argument{"focal lengths must be positive"};
    }
}

Pixel Camera::project(const Point3& point_camera_m) const {
    if (!std::isfinite(point_camera_m.x_m) ||
        !std::isfinite(point_camera_m.y_m) ||
        !std::isfinite(point_camera_m.z_m)) {
        throw std::invalid_argument{"point coordinates must be finite"};
    }
    if (point_camera_m.z_m <= 0.0) {
        throw std::domain_error{"point must have positive camera-frame depth"};
    }

    return Pixel{
        fx_px_ * point_camera_m.x_m / point_camera_m.z_m + cx_px_,
        fy_px_ * point_camera_m.y_m / point_camera_m.z_m + cy_px_,
    };
}

}  // namespace day018
