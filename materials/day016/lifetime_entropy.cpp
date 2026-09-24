#include <cmath>
#include <cstddef>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

struct CopyCounted {
    double value;
    static int copies;

    explicit CopyCounted(double value_in) : value{value_in} {}
    CopyCounted(const CopyCounted& other) : value{other.value} { ++copies; }
    CopyCounted& operator=(const CopyCounted& other) {
        value = other.value;
        ++copies;
        return *this;
    }
};

int CopyCounted::copies = 0;

double sum_by_value(std::vector<CopyCounted> values) {
    double sum = 0.0;
    for (const CopyCounted& item : values) sum += item.value;
    return sum;
}

double sum_by_const_reference(const std::vector<CopyCounted>& values) {
    double sum = 0.0;
    for (const CopyCounted& item : values) sum += item.value;
    return sum;
}

void add_offset(double& value_m, double offset_m) {
    if (!std::isfinite(value_m) || !std::isfinite(offset_m)) {
        throw std::invalid_argument("value and offset must be finite");
    }
    value_m += offset_m;
}

double mean_from_observer(const double* data, std::size_t size) {
    if (data == nullptr) throw std::invalid_argument("data pointer must not be null");
    if (size == 0) throw std::invalid_argument("size must be positive");
    double sum = 0.0;
    for (std::size_t index = 0; index < size; ++index) {
        if (!std::isfinite(data[index]) || data[index] < 0.0) {
            throw std::invalid_argument("ranges must be finite and non-negative");
        }
        sum += data[index];
    }
    return sum / static_cast<double>(size);
}

const double* element_observer(const std::vector<double>& values, std::size_t index) {
    if (index >= values.size()) throw std::out_of_range("index outside vector");
    return &values[index];
}

double binary_entropy_bits(double probability) {
    if (!std::isfinite(probability) || probability < 0.0 || probability > 1.0) {
        throw std::invalid_argument("probability must satisfy 0 <= p <= 1");
    }
    if (probability == 0.0 || probability == 1.0) return 0.0;
    return -probability * std::log2(probability)
           - (1.0 - probability) * std::log2(1.0 - probability);
}

bool close(double left, double right, double tolerance = 1e-12) {
    return std::abs(left - right) <= tolerance;
}

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error("self-test failed: " + message);
}

void write_entropy_csv(const std::string& path) {
    std::ofstream output{path};
    if (!output) throw std::runtime_error("could not open CSV output");
    output << "p,entropy_bits\n" << std::fixed << std::setprecision(6);
    for (int step = 0; step <= 10; ++step) {
        const double probability = static_cast<double>(step) / 10.0;
        output << probability << ',' << binary_entropy_bits(probability) << '\n';
    }
    if (!output) throw std::runtime_error("failed while writing CSV output");
    std::cout << "CSV_OK:path=" << path << ",rows=11\n";
}

void run_self_tests() {
    std::vector<CopyCounted> values;
    values.reserve(4);
    values.emplace_back(0.8);
    values.emplace_back(1.0);
    values.emplace_back(1.2);
    values.emplace_back(1.4);

    CopyCounted::copies = 0;
    require(close(sum_by_value(values), 4.4), "sum by value");
    require(CopyCounted::copies == 4, "four copies for pass by value");
    CopyCounted::copies = 0;
    require(close(sum_by_const_reference(values), 4.4), "sum by const reference");
    require(CopyCounted::copies == 0, "no copies for pass by const reference");

    double distance_m = 2.0;
    double& alias_m = distance_m;
    add_offset(alias_m, 0.5);
    require(close(distance_m, 2.5), "reference aliases original");

    const std::vector<double> ranges_m{0.8, 1.0, 1.2, 1.4};
    require(close(mean_from_observer(ranges_m.data(), ranges_m.size()), 1.1), "pointer mean");
    require(close(*element_observer(ranges_m, 2), 1.2), "element observer");
    require(close(binary_entropy_bits(0.0), 0.0), "entropy p=0");
    require(close(binary_entropy_bits(0.5), 1.0), "entropy p=0.5");
    require(close(binary_entropy_bits(1.0), 0.0), "entropy p=1");
    require(close(binary_entropy_bits(0.25), binary_entropy_bits(0.75)), "entropy symmetry");

    bool rejected_null = false;
    try { static_cast<void>(mean_from_observer(nullptr, 4)); }
    catch (const std::invalid_argument&) { rejected_null = true; }
    require(rejected_null, "null observer rejection");

    bool rejected_index = false;
    try { static_cast<void>(element_observer(ranges_m, ranges_m.size())); }
    catch (const std::out_of_range&) { rejected_index = true; }
    require(rejected_index, "out-of-range rejection");

    bool rejected_probability = false;
    try { static_cast<void>(binary_entropy_bits(1.1)); }
    catch (const std::invalid_argument&) { rejected_probability = true; }
    require(rejected_probability, "invalid probability rejection");

    bool rejected_range = false;
    try {
        const double bad_ranges[]{1.0, -0.1};
        static_cast<void>(mean_from_observer(bad_ranges, 2));
    } catch (const std::invalid_argument&) { rejected_range = true; }
    require(rejected_range, "invalid range rejection");

    std::cout << "SELF_TEST_OK: 14 checks\n";
}

void print_help() {
    std::cout << "Usage: lifetime_entropy [--self-test|--write-csv PATH|--help]\n";
}

int main(int argc, char* argv[]) {
    try {
        if (argc == 2 && std::string{argv[1]} == "--self-test") {
            run_self_tests();
            return 0;
        }
        if (argc == 3 && std::string{argv[1]} == "--write-csv") {
            write_entropy_csv(argv[2]);
            return 0;
        }
        if (argc == 2 && std::string{argv[1]} == "--help") {
            print_help();
            return 0;
        }
        if (argc != 1) {
            std::cerr << "error: unknown or incomplete argument\n";
            print_help();
            return 2;
        }

        std::vector<CopyCounted> values;
        values.reserve(4);
        for (const double value : {0.8, 1.0, 1.2, 1.4}) values.emplace_back(value);

        CopyCounted::copies = 0;
        const double value_sum = sum_by_value(values);
        const int value_copies = CopyCounted::copies;
        CopyCounted::copies = 0;
        const double reference_sum = sum_by_const_reference(values);
        const int reference_copies = CopyCounted::copies;

        double distance_m = 2.0;
        double& alias_m = distance_m;
        add_offset(alias_m, 0.5);

        const std::vector<double> ranges_m{0.8, 1.0, 1.2, 1.4};
        const double* observed = element_observer(ranges_m, 2);

        std::cout << std::fixed << std::setprecision(6);
        std::cout << "contract=ranges_m:(N,),frame=sensor,unit=m,N>0,finite,nonnegative\n";
        std::cout << "copy_demo:by_value_copies=" << value_copies
                  << ",by_const_ref_copies=" << reference_copies
                  << ",sum_value=" << value_sum
                  << ",sum_ref=" << reference_sum << '\n';
        std::cout << "alias_demo:original_m=" << distance_m
                  << ",alias_m=" << alias_m << '\n';
        std::cout << "pointer_demo:index=2,value_m=" << *observed
                  << ",mean_m=" << mean_from_observer(ranges_m.data(), ranges_m.size()) << '\n';
        for (const double probability : {0.0, 0.25, 0.5, 0.75, 1.0}) {
            std::cout << "entropy:p=" << probability
                      << ",bits=" << binary_entropy_bits(probability) << '\n';
        }
        std::cout << "SUCCESS: references, observers, and Bernoulli entropy validated\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
