#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>

#include "bmp.h"
#include "color.h"
#include "coordinate.h"
#include "fib_test.h"
#include "functional.h"
#include "images.h"

namespace Rentib {

constexpr uint32_t width = 800;
constexpr uint32_t height = 600;
auto rgb(int r, int g, int b) { return Color(b, g, r); }

const Color bg = rgb(40, 40, 40);
const Color fg = rgb(187, 172, 146);
const Color black = rgb(39, 39, 39);
const Color yellow = rgb(213, 152, 32);
const Color red = rgb(202, 35, 28);
const Color green = rgb(151, 150, 25);
const Color blue = rgb(68, 132, 135);
const Color pink = rgb(175, 97, 133);
const Color aqua = rgb(103, 156, 105);

auto getRegion(Image img) {
  return [=](const Point p) { return img(p) == bg ? false : true; };
}

void test() {}

}  // namespace Rentib

int main() {
  const uint32_t width = 400;
  const uint32_t height = 300;
  const Region rc = circle(Point(50., 100.), 10., true, false);
  const Image vs = vertical_stripe(100, Colors::Vermilion, Colors::blue);
  const Blend cb = constant<Fraction>(.42);

  create_BMP("test_images/constant.bmp", width, height, constant(Colors::Vermilion));
  create_BMP("test_images/rotate.bmp", width, height, rotate(vs, M_PI / 4.));
  create_BMP("test_images/translate.bmp", width, height, translate(vs, Vector(100., 0.)));
  create_BMP("test_images/scale.bmp", width, height, scale(vs, 2.));
  create_BMP("test_images/circle.bmp", width, height,
             circle(Point(50., 100.), 10., Colors::Vermilion, Colors::blue));
  create_BMP("test_images/checker.bmp", width, height,
             checker(10., Colors::Vermilion, Colors::blue));
  create_BMP("test_images/polar_checker.bmp", width, height,
             polar_checker(10., 4, Colors::Vermilion, Colors::blue));
  create_BMP("test_images/rings.bmp", width, height,
             rings(Point(50., 100.), 10., Colors::Vermilion, Colors::blue));
  create_BMP("test_images/vertical_stripe.bmp", width, height, vs);
  create_BMP("test_images/cond.bmp", width, height,
             cond(rc, constant(Colors::Vermilion), constant(Colors::blue)));
  create_BMP("test_images/lerp.bmp", width, height,
             lerp(cb, constant(Colors::blue), constant(Colors::white)));
  create_BMP("test_images/dark_vs.bmp", width, height, darken(vs, cb));
  create_BMP("test_images/light_vs.bmp", width, height, lighten(vs, cb));

  assert(compose()(42) == 42);
  assert(compose([](auto x) { return x + 1; }, [](auto x) { return x * x; })(1) == 4);

  const auto h1 = [](auto a, auto b) {
    auto g = a * b;
    return g;
  };
  const auto h2 = [](auto a, auto b) {
    auto g = a + b;
    return g;
  };
  const auto f1 = [](auto p) {
    auto a = p;
    return a;
  };
  const auto f2 = [](auto p) {
    auto b = p;
    return b;
  };
  assert(lift(h1, f1, f2)(42) == 42 * 42);
  assert(lift(h2, f1, f2)(42) == 42 + 42);

  Fibonacci::test();
}
