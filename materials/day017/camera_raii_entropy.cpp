#include <array>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

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
    Camera(double fx_px, double fy_px, double cx_px, double cy_px,
           std::array<double, 9> rotation_cw,
           std::array<double, 3> translation_cw_m)
        : fx_px_{fx_px}, fy_px_{fy_px}, cx_px_{cx_px}, cy_px_{cy_px},
          rotation_cw_{rotation_cw}, translation_cw_m_{translation_cw_m} {
        validate();
    }

    Point3 world_to_camera(const Point3& world) const {
        require_finite(world);
        return {
            rotation_cw_[0] * world.x_m + rotation_cw_[1] * world.y_m
                + rotation_cw_[2] * world.z_m + translation_cw_m_[0],
            rotation_cw_[3] * world.x_m + rotation_cw_[4] * world.y_m
                + rotation_cw_[5] * world.z_m + translation_cw_m_[1],
            rotation_cw_[6] * world.x_m + rotation_cw_[7] * world.y_m
                + rotation_cw_[8] * world.z_m + translation_cw_m_[2],
        };
    }

    Pixel project(const Point3& world) const {
        const Point3 camera = world_to_camera(world);
        if (camera.z_m <= 0.0) {
            throw std::invalid_argument("point must have positive camera-frame depth");
        }
        return {fx_px_ * camera.x_m / camera.z_m + cx_px_,
                fy_px_ * camera.y_m / camera.z_m + cy_px_};
    }

    double principal_u_px() const { return cx_px_; }

private:
    static void require_finite(const Point3& point) {
        if (!std::isfinite(point.x_m) || !std::isfinite(point.y_m)
            || !std::isfinite(point.z_m)) {
            throw std::invalid_argument("point coordinates must be finite");
        }
    }

    void validate() const {
        for (const double value : rotation_cw_) {
            if (!std::isfinite(value)) throw std::invalid_argument("rotation must be finite");
        }
        for (const double value : translation_cw_m_) {
            if (!std::isfinite(value)) throw std::invalid_argument("translation must be finite");
        }
        if (!std::isfinite(fx_px_) || !std::isfinite(fy_px_)
            || !std::isfinite(cx_px_) || !std::isfinite(cy_px_)
            || fx_px_ <= 0.0 || fy_px_ <= 0.0) {
            throw std::invalid_argument("intrinsics must be finite and focal lengths positive");
        }
        for (int row = 0; row < 3; ++row) {
            double norm = 0.0;
            for (int column = 0; column < 3; ++column) {
                const double value = rotation_cw_[3 * row + column];
                norm += value * value;
            }
            if (std::abs(norm - 1.0) > 1e-9) {
                throw std::invalid_argument("rotation rows must have unit norm");
            }
        }
        for (int left = 0; left < 3; ++left) {
            for (int right = left + 1; right < 3; ++right) {
                double dot = 0.0;
                for (int column = 0; column < 3; ++column) {
                    dot += rotation_cw_[3 * left + column]
                           * rotation_cw_[3 * right + column];
                }
                if (std::abs(dot) > 1e-9) {
                    throw std::invalid_argument("rotation rows must be orthogonal");
                }
            }
        }
        const double determinant =
            rotation_cw_[0] * (rotation_cw_[4] * rotation_cw_[8]
                               - rotation_cw_[5] * rotation_cw_[7])
            - rotation_cw_[1] * (rotation_cw_[3] * rotation_cw_[8]
                                 - rotation_cw_[5] * rotation_cw_[6])
            + rotation_cw_[2] * (rotation_cw_[3] * rotation_cw_[7]
                                 - rotation_cw_[4] * rotation_cw_[6]);
        if (std::abs(determinant - 1.0) > 1e-9) {
            throw std::invalid_argument("rotation determinant must be +1");
        }
    }

    double fx_px_;
    double fy_px_;
    double cx_px_;
    double cy_px_;
    std::array<double, 9> rotation_cw_;
    std::array<double, 3> translation_cw_m_;
};

class ProjectionReport {
public:
    explicit ProjectionReport(const std::string& path) : output_{path} {
        if (!output_) throw std::runtime_error("could not open report output");
        ++active_reports_;
        output_ << "x_w_m,y_w_m,z_w_m,x_c_m,y_c_m,z_c_m,u_px,v_px,depth_bin,side\n";
    }

    ProjectionReport(const ProjectionReport&) = delete;
    ProjectionReport& operator=(const ProjectionReport&) = delete;

    ~ProjectionReport() {
        output_ << "# closed_by_raii\n";
        --active_reports_;
    }

    void append(const Point3& world, const Point3& camera, const Pixel& pixel,
                int depth_bin, int side) {
        output_ << std::fixed << std::setprecision(6)
                << world.x_m << ',' << world.y_m << ',' << world.z_m << ','
                << camera.x_m << ',' << camera.y_m << ',' << camera.z_m << ','
                << pixel.u_px << ',' << pixel.v_px << ',' << depth_bin << ',' << side << '\n';
        if (!output_) throw std::runtime_error("failed while writing report");
    }

    static int active_reports() { return active_reports_; }

private:
    std::ofstream output_;
    static int active_reports_;
};

int ProjectionReport::active_reports_ = 0;

struct EntropySummary {
    std::array<std::size_t, 4> counts{};
    double h_x_bits{};
    double h_y_bits{};
    double h_xy_bits{};
    double h_y_given_x_bits{};
};

double entropy_bits(const std::vector<double>& probabilities) {
    double total = 0.0;
    double entropy = 0.0;
    for (const double probability : probabilities) {
        if (!std::isfinite(probability) || probability < 0.0 || probability > 1.0) {
            throw std::invalid_argument("probabilities must lie in [0,1]");
        }
        total += probability;
        if (probability > 0.0) entropy -= probability * std::log2(probability);
    }
    if (std::abs(total - 1.0) > 1e-12) {
        throw std::invalid_argument("probabilities must sum to one");
    }
    return entropy;
}

EntropySummary summarize(const std::vector<int>& depth_bins,
                         const std::vector<int>& sides) {
    if (depth_bins.empty() || depth_bins.size() != sides.size()) {
        throw std::invalid_argument("labels must be non-empty and paired");
    }
    EntropySummary result;
    for (std::size_t index = 0; index < depth_bins.size(); ++index) {
        const int x = depth_bins[index];
        const int y = sides[index];
        if ((x != 0 && x != 1) || (y != 0 && y != 1)) {
            throw std::invalid_argument("labels must be binary");
        }
        ++result.counts[static_cast<std::size_t>(2 * x + y)];
    }
    const double n = static_cast<double>(depth_bins.size());
    const std::vector<double> joint{
        result.counts[0] / n, result.counts[1] / n,
        result.counts[2] / n, result.counts[3] / n,
    };
    const std::vector<double> px{joint[0] + joint[1], joint[2] + joint[3]};
    const std::vector<double> py{joint[0] + joint[2], joint[1] + joint[3]};
    result.h_x_bits = entropy_bits(px);
    result.h_y_bits = entropy_bits(py);
    result.h_xy_bits = entropy_bits(joint);
    result.h_y_given_x_bits = result.h_xy_bits - result.h_x_bits;
    return result;
}

bool close(double left, double right, double tolerance = 1e-9) {
    return std::abs(left - right) <= tolerance;
}

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error("self-test failed: " + message);
}

std::unique_ptr<Camera> make_teaching_camera() {
    return std::make_unique<Camera>(
        100.0, 100.0, 320.0, 240.0,
        std::array<double, 9>{1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0},
        std::array<double, 3>{0.0, 0.0, 0.0});
}

const std::vector<Point3>& teaching_points() {
    static const std::vector<Point3> points{
        {-1.0, -0.4, 2.0}, {-0.8, 0.2, 2.0}, {-0.4, 0.6, 2.0}, {0.5, -0.2, 2.0},
        {-1.0, -0.8, 4.0}, {0.4, -0.4, 4.0}, {0.8, 0.4, 4.0}, {1.2, 0.8, 4.0},
    };
    return points;
}

EntropySummary run_dataset(const Camera& camera, const std::string* report_path) {
    std::vector<int> depth_bins;
    std::vector<int> sides;
    std::unique_ptr<ProjectionReport> report;
    if (report_path != nullptr) report = std::make_unique<ProjectionReport>(*report_path);
    for (const Point3& world : teaching_points()) {
        const Point3 camera_point = camera.world_to_camera(world);
        const Pixel pixel = camera.project(world);
        const int depth_bin = camera_point.z_m < 3.0 ? 0 : 1;
        const int side = pixel.u_px < camera.principal_u_px() ? 0 : 1;
        depth_bins.push_back(depth_bin);
        sides.push_back(side);
        if (report) report->append(world, camera_point, pixel, depth_bin, side);
    }
    return summarize(depth_bins, sides);
}

void run_self_tests() {
    auto camera = make_teaching_camera();
    const Pixel center = camera->project({0.0, 0.0, 2.0});
    require(close(center.u_px, 320.0) && close(center.v_px, 240.0), "principal point");
    const Pixel corner = camera->project({-1.0, -1.0, 2.0});
    require(close(corner.u_px, 270.0) && close(corner.v_px, 190.0), "known projection");
    const Point3 transformed = camera->world_to_camera({1.0, 2.0, 3.0});
    require(close(transformed.x_m, 1.0), "identity x");
    require(close(transformed.y_m, 2.0), "identity y");
    require(close(transformed.z_m, 3.0), "identity z");
    require(ProjectionReport::active_reports() == 0, "no report before scope");
    const EntropySummary summary = run_dataset(*camera, nullptr);
    require(summary.counts == std::array<std::size_t, 4>{3, 1, 1, 3}, "joint counts");
    require(close(summary.h_x_bits, 1.0), "H(X)");
    require(close(summary.h_y_bits, 1.0), "H(Y)");
    require(close(summary.h_xy_bits, 1.811278124459133), "H(X,Y)");
    require(close(summary.h_y_given_x_bits, 0.811278124459133), "H(Y|X)");
    require(close(summary.h_xy_bits,
                  summary.h_x_bits + summary.h_y_given_x_bits), "chain rule");
    require(close(entropy_bits({1.0, 0.0}), 0.0), "zero-mass convention");
    bool rejected_depth = false;
    try { static_cast<void>(camera->project({0.0, 0.0, 0.0})); }
    catch (const std::invalid_argument&) { rejected_depth = true; }
    require(rejected_depth, "non-positive depth rejection");
    bool rejected_intrinsics = false;
    try {
        Camera invalid{0.0, 100.0, 320.0, 240.0,
                       {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0},
                       {0.0, 0.0, 0.0}};
        static_cast<void>(invalid);
    } catch (const std::invalid_argument&) { rejected_intrinsics = true; }
    require(rejected_intrinsics, "invalid focal length rejection");
    bool rejected_probabilities = false;
    try { static_cast<void>(entropy_bits({0.6, 0.6})); }
    catch (const std::invalid_argument&) { rejected_probabilities = true; }
    require(rejected_probabilities, "probability sum rejection");
    std::cout << "SELF_TEST_OK: 16 checks\n";
}

void print_summary(const EntropySummary& summary) {
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "joint_counts:x0y0=" << summary.counts[0]
              << ",x0y1=" << summary.counts[1]
              << ",x1y0=" << summary.counts[2]
              << ",x1y1=" << summary.counts[3] << '\n';
    std::cout << "entropy:h_x_bits=" << summary.h_x_bits
              << ",h_y_bits=" << summary.h_y_bits
              << ",h_xy_bits=" << summary.h_xy_bits
              << ",h_y_given_x_bits=" << summary.h_y_given_x_bits << '\n';
    std::cout << "chain_rule:error_bits="
              << std::abs(summary.h_xy_bits
                          - summary.h_x_bits - summary.h_y_given_x_bits) << '\n';
}

void print_help() {
    std::cout << "Usage: camera_raii_entropy [--self-test|--write-report PATH|--help]\n";
}

int main(int argc, char* argv[]) {
    try {
        if (argc == 2 && std::string{argv[1]} == "--self-test") {
            run_self_tests();
            return 0;
        }
        if (argc == 2 && std::string{argv[1]} == "--help") {
            print_help();
            return 0;
        }
        const bool write_report = argc == 3 && std::string{argv[1]} == "--write-report";
        if (argc != 1 && !write_report) {
            std::cerr << "error: unknown or incomplete argument\n";
            print_help();
            return 2;
        }

        auto camera = make_teaching_camera();
        std::cout << "contract=points_world_m:(N,3),R_cw:(3,3),t_cw_m:(3,),pixels:(N,2)\n";
        std::cout << "ownership=unique_ptr<Camera>,exclusive=1,manual_delete=0\n";
        const Pixel known = camera->project({-1.0, -1.0, 2.0});
        std::cout << std::fixed << std::setprecision(6)
                  << "projection:world_m=(-1,-1,2),pixel_px=("
                  << known.u_px << ',' << known.v_px << "),expected_px=(270,190)\n";

        EntropySummary summary;
        if (write_report) {
            {
                ProjectionReport report{argv[2]};
                std::vector<int> depth_bins;
                std::vector<int> sides;
                for (const Point3& world : teaching_points()) {
                    const Point3 camera_point = camera->world_to_camera(world);
                    const Pixel pixel = camera->project(world);
                    const int depth_bin = camera_point.z_m < 3.0 ? 0 : 1;
                    const int side = pixel.u_px < camera->principal_u_px() ? 0 : 1;
                    depth_bins.push_back(depth_bin);
                    sides.push_back(side);
                    report.append(world, camera_point, pixel, depth_bin, side);
                }
                summary = summarize(depth_bins, sides);
                std::cout << "raii_scope:active_inside=" << ProjectionReport::active_reports() << '\n';
            }
            std::cout << "raii_scope:active_after=" << ProjectionReport::active_reports() << '\n';
            std::cout << "REPORT_OK:path=" << argv[2] << ",rows=8\n";
        } else {
            summary = run_dataset(*camera, nullptr);
        }
        print_summary(summary);
        std::cout << "SUCCESS: Camera invariants, RAII ownership, projection, and entropy validated\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
