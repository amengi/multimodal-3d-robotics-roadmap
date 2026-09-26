#include "camera.hpp"

#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string_view>

namespace {

bool close(double left, double right, double tolerance = 1e-12) {
    return std::abs(left - right) <= tolerance;
}

int self_test() {
    int checks = 0;
    const day018::Camera camera{100.0, 100.0, 320.0, 240.0};
    const auto pixel = camera.project(day018::Point3{-1.0, -1.0, 2.0});
    if (!close(pixel.u_px, 270.0) || !close(pixel.v_px, 190.0)) {
        throw std::runtime_error{"known projection failed"};
    }
    checks += 2;

    const auto principal = camera.project(day018::Point3{0.0, 0.0, 3.0});
    if (!close(principal.u_px, 320.0) || !close(principal.v_px, 240.0)) {
        throw std::runtime_error{"principal-point projection failed"};
    }
    checks += 2;

    try {
        static_cast<void>(camera.project(day018::Point3{0.0, 0.0, 0.0}));
        throw std::runtime_error{"zero depth was accepted"};
    } catch (const std::domain_error&) {
        ++checks;
    }

    try {
        static_cast<void>(day018::Camera{0.0, 100.0, 320.0, 240.0});
        throw std::runtime_error{"zero focal length was accepted"};
    } catch (const std::invalid_argument&) {
        ++checks;
    }

    if (checks != 6) {
        throw std::runtime_error{"unexpected self-test count"};
    }
    std::cout << "SELF_TEST_OK: " << checks << " checks\n";
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc == 2 && std::string_view{argv[1]} == "--self-test") {
            return self_test();
        }
        if (argc != 1) {
            std::cerr << "usage: camera_app [--self-test]\n";
            return 2;
        }

        const day018::Camera camera{100.0, 100.0, 320.0, 240.0};
        const auto pixel = camera.project(day018::Point3{-1.0, -1.0, 2.0});
        std::cout << std::fixed << std::setprecision(6);
        std::cout << "translation_units=2\n";
        std::cout << "projection_px=(" << pixel.u_px << ',' << pixel.v_px << ")\n";
        std::cout << "geometry_error_px="
                  << std::hypot(pixel.u_px - 270.0, pixel.v_px - 190.0)
                  << "\nSUCCESS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "ERROR: " << error.what() << '\n';
        return 1;
    }
}
