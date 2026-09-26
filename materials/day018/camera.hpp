#pragma once

namespace day018 {

struct Point3 {
    double x_m;
    double y_m;
    double z_m;
};

struct Pixel {
    double u_px;
    double v_px;
};

class Camera {
public:
    Camera(double fx_px, double fy_px, double cx_px, double cy_px);

    [[nodiscard]] Pixel project(const Point3& point_camera_m) const;

private:
    double fx_px_;
    double fy_px_;
    double cx_px_;
    double cy_px_;
};

}  // namespace day018
