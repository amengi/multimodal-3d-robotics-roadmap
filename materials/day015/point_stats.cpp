#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

struct Point2D {
    double x;
    double y;
};

struct PointStats {
    std::size_t count;
    Point2D centroid;
    Point2D minimum;
    Point2D maximum;
};

PointStats compute_stats(const std::vector<Point2D>& points) {
    if (points.empty()) {
        throw std::invalid_argument("points must not be empty");
    }

    double sum_x = 0.0;
    double sum_y = 0.0;
    Point2D minimum = points.front();
    Point2D maximum = points.front();

    for (const Point2D& point : points) {
        if (!std::isfinite(point.x) || !std::isfinite(point.y)) {
            throw std::invalid_argument("point coordinates must be finite");
        }
        sum_x += point.x;
        sum_y += point.y;
        if (point.x < minimum.x) minimum.x = point.x;
        if (point.y < minimum.y) minimum.y = point.y;
        if (point.x > maximum.x) maximum.x = point.x;
        if (point.y > maximum.y) maximum.y = point.y;
    }

    const double count = static_cast<double>(points.size());
    return PointStats{
        points.size(),
        Point2D{sum_x / count, sum_y / count},
        minimum,
        maximum,
    };
}

double self_information(double probability, double log_base) {
    if (!(probability > 0.0 && probability <= 1.0)) {
        throw std::invalid_argument("probability must satisfy 0 < p <= 1");
    }
    if (!(log_base > 1.0)) {
        throw std::invalid_argument("log base must be greater than 1");
    }
    if (probability == 1.0) {
        return 0.0;
    }
    return -std::log(probability) / std::log(log_base);
}

bool close(double left, double right, double tolerance = 1e-12) {
    return std::abs(left - right) <= tolerance;
}

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error("self-test failed: " + message);
    }
}

void run_self_tests() {
    const std::vector<Point2D> points{{1.0, 2.0}, {3.0, 4.0}, {-1.0, 0.0}, {1.0, -2.0}};
    const PointStats stats = compute_stats(points);
    require(stats.count == 4, "count");
    require(close(stats.centroid.x, 1.0) && close(stats.centroid.y, 1.0), "centroid");
    require(close(stats.minimum.x, -1.0) && close(stats.minimum.y, -2.0), "minimum");
    require(close(stats.maximum.x, 3.0) && close(stats.maximum.y, 4.0), "maximum");
    require(close(self_information(1.0, 2.0), 0.0), "certain event");
    require(close(self_information(0.5, 2.0), 1.0), "fair binary outcome");

    bool rejected_empty = false;
    try {
        static_cast<void>(compute_stats({}));
    } catch (const std::invalid_argument&) {
        rejected_empty = true;
    }
    require(rejected_empty, "empty input rejection");

    bool rejected_probability = false;
    try {
        static_cast<void>(self_information(0.0, 2.0));
    } catch (const std::invalid_argument&) {
        rejected_probability = true;
    }
    require(rejected_probability, "zero probability rejection");

    bool rejected_log_base = false;
    try {
        static_cast<void>(self_information(0.5, 1.0));
    } catch (const std::invalid_argument&) {
        rejected_log_base = true;
    }
    require(rejected_log_base, "invalid log-base rejection");

    bool rejected_nonfinite = false;
    try {
        static_cast<void>(compute_stats({Point2D{std::numeric_limits<double>::infinity(), 0.0}}));
    } catch (const std::invalid_argument&) {
        rejected_nonfinite = true;
    }
    require(rejected_nonfinite, "non-finite point rejection");

    std::cout << "SELF_TEST_OK: 10 checks\n";
}

void print_help() {
    std::cout << "Usage: point_stats [--self-test|--help]\n";
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
        if (argc != 1) {
            std::cerr << "error: unknown argument\n";
            print_help();
            return 2;
        }

        const std::vector<Point2D> points{{1.0, 2.0}, {3.0, 4.0}, {-1.0, 0.0}, {1.0, -2.0}};
        const PointStats stats = compute_stats(points);

        std::cout << std::fixed << std::setprecision(6);
        std::cout << "contract=points_m:(N,2),frame=world,unit=m\n";
        std::cout << "count=" << stats.count << '\n';
        std::cout << "centroid_m=" << stats.centroid.x << ',' << stats.centroid.y << '\n';
        std::cout << "aabb_min_m=" << stats.minimum.x << ',' << stats.minimum.y << '\n';
        std::cout << "aabb_max_m=" << stats.maximum.x << ',' << stats.maximum.y << '\n';

        const std::vector<double> probabilities{1.0, 0.5, 0.25, 0.125};
        for (const double probability : probabilities) {
            std::cout << "self_info:p=" << probability
                      << ",bits=" << self_information(probability, 2.0)
                      << ",nats=" << self_information(probability, std::exp(1.0)) << '\n';
        }
        std::cout << "SUCCESS: C++ point statistics and self-information validated\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
