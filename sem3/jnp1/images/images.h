#ifndef IMAGES_H_
#define IMAGES_H_

#include <cmath>
#include <functional>

#include "color.h"
#include "coordinate.h"
#include "functional.h"

using Fraction = double;

template <typename T>
using Base_image = std::function<T(const Point)>;

using Region = Base_image<bool>;
using Image = Base_image<Color>;
using Blend = Base_image<Fraction>;

namespace Detail {
inline auto make_polar(const Point p) { return p.is_polar ? p : to_polar(p); }
inline auto make_normal(const Point p) { return p.is_polar ? from_polar(p) : p; }
}  // namespace Detail

template <typename T>
Base_image<T> constant(T t) {
  return [=]([[maybe_unused]] const Point p) { return t; };
}

template <typename T>
Base_image<T> rotate(Base_image<T> img, double phi) {
  return compose(Detail::make_polar, [=](const Point p) {
    return img({p.first, p.second - phi, true});
  });
}

template <typename T>
Base_image<T> translate(Base_image<T> img, Vector v) {
  return compose(Detail::make_normal, [=](const Point p) {
    return img({p.first - v.first, p.second - v.second, false});
  });
}

template <typename T>
Base_image<T> scale(Base_image<T> img, double s) {
  return compose(Detail::make_normal, [=](const Point p) {
    return img({p.first / s, p.second / s, false});
  });
}

template <typename T>
Base_image<T> circle(Point q, double r, T inner, T outer) {
  return [=](const Point pt) {
    return distance(Detail::make_normal(pt), Detail::make_normal(q)) <= r ? inner : outer;
  };
}

template <typename T>
Base_image<T> checker(double d, T this_way, T that_way) {
  return compose(Detail::make_normal, [=](const Point p) {
    return (static_cast<int>(std::floor(p.first / d)) +
            static_cast<int>(std::floor(p.second / d))) % 2
               ? that_way : this_way;
  });
}

template <typename T>
Base_image<T> polar_checker(double d, int n, T this_way, T that_way) {
  return compose(Detail::make_polar, [=](const Point p) {
    return Point{p.first, p.second * n * d / (2 * M_PI), false};
  }, checker(d, this_way, that_way));
}

template <typename T>
Base_image<T> rings(Point q, double d, T this_way, T that_way) {
  return [=](const Point pt) {
    return static_cast<int>(distance(Detail::make_normal(pt), Detail::make_normal(q)) / d) & 1
    ? that_way : this_way;
  };
}

template <typename T>
Base_image<T> vertical_stripe(double d, T this_way, T that_way) {
  return compose(Detail::make_normal, [=](const Point p) {
    return std::abs(p.first) * 2 <= d ? this_way : that_way;
  });
}

inline Image cond(Region region, Image this_way, Image that_way) {
  return [=](const Point pt) { return region(pt) ? this_way(pt) : that_way(pt); };
}

inline Image lerp(Blend blend, Image this_way, Image that_way) {
  return lift([=](Color a, Color b, Fraction w) { return a.weighted_mean(b, w); },
                  this_way, that_way, blend);
}

inline Image darken(Image img, Blend blend) { return lerp(blend, img, constant(Colors::black)); }

inline Image lighten(Image img, Blend blend) { return lerp(blend, img, constant(Colors::white)); }

#endif  // IMAGES_H_
