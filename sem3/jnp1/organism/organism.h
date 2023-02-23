#ifndef __ORGANISM_H__
#define __ORGANISM_H__

#include <compare>
#include <concepts>
#include <cstdint>
#include <optional>
#include <tuple>
#include <type_traits>
#include <utility>

namespace detail {
template<typename Organism1, typename Organism2>
using encounter_result_t =
  std::tuple<Organism1, Organism2, std::optional<Organism1>>;
} // namespace detail

template<typename species_t, bool can_eat_meat, bool can_eat_plants>
  requires std::equality_comparable<species_t>
class Organism
{
public:
  static constexpr bool is_animal = can_eat_meat || can_eat_plants;

  constexpr Organism(const species_t& species, std::uint64_t vitality) noexcept
    : species_{ species }
    , vitality_{ vitality }
  {
  }

  constexpr std::optional<Organism> breed(const Organism& other) const noexcept
    requires(is_animal)
  {
    if (get_species() == other.get_species())
      /* return { { get_species(), (get_vitality() + other.get_vitality()) / 2 } }; */
      return { { get_species(), get_vitality() / 2 + other.get_vitality() / 2
                              + ((get_vitality() & 1) + (other.get_vitality() & 1)) / 2 } };
    else
      return {};
  }

  constexpr Organism die() && noexcept
  {
    return { std::move(*this).get_species(), 0 };
  }

  template<bool can_eat_meat2, bool can_eat_plants2>
    requires(can_eat_meat && (can_eat_meat2 || can_eat_plants2))
  constexpr detail::encounter_result_t<
    Organism,
    Organism<species_t, can_eat_meat2, can_eat_plants2>>
  get_killed(
    Organism<species_t, can_eat_meat2, can_eat_plants2> other) && noexcept
  {
    auto [result_other, result_this, discard] =
      std::move(other).kill(std::move(*this));
    return { std::move(result_this), std::move(result_other), {} };
  }

  constexpr detail::encounter_result_t<Organism,
                                       Organism<species_t, false, false>>
  graze(Organism<species_t, false, false> other) && noexcept
    requires(can_eat_plants)
  {
    return { { std::move(*this).get_species(),
               get_vitality() + other.get_vitality() },
             std::move(other).die(),
             {} };
  }

  template<bool can_eat_meat2, bool can_eat_plants2>
  constexpr detail::encounter_result_t<
    Organism,
    Organism<species_t, can_eat_meat2, can_eat_plants2>>
  ignore(Organism<species_t, can_eat_meat2, can_eat_plants2> other) && noexcept
  {
    return { std::move(*this), std::move(other), {} };
  }

  constexpr bool is_dead() const noexcept { return get_vitality() == 0; }

  template<bool can_eat_meat2, bool can_eat_plants2>
    requires(can_eat_meat && (can_eat_meat2 || can_eat_plants2))
  constexpr detail::encounter_result_t<
    Organism,
    Organism<species_t, can_eat_meat2, can_eat_plants2>>
  kill(Organism<species_t, can_eat_meat2, can_eat_plants2> other) && noexcept
  {
    return { { std::move(*this).get_species(),
               get_vitality() + other.get_vitality() / 2 },
             std::move(other).die(),
             {} };
  }

  constexpr const species_t& get_species() const& noexcept { return species_; }

  constexpr species_t get_species() && noexcept { return std::move(species_); }

  constexpr std::uint64_t get_vitality() const noexcept { return vitality_; }

private:
  species_t species_;
  uint64_t vitality_;
};

template<typename species_t>
using Carnivore = Organism<species_t, true, false>;

template<typename species_t>
using Omnivore = Organism<species_t, true, true>;

template<typename species_t>
using Herbivore = Organism<species_t, false, true>;

template<typename species_t>
using Plant = Organism<species_t, false, false>;

namespace detail {
template<typename species_t,
         bool sp1_eats_m,
         bool sp1_eats_p,
         bool sp2_eats_m,
         bool sp2_eats_p>
struct encounter_helper
{
  static constexpr bool possible = true;

  static constexpr encounter_result_t<
    Organism<species_t, sp1_eats_m, sp1_eats_p>,
    Organism<species_t, sp2_eats_m, sp2_eats_p>>
  encounter(Organism<species_t, sp1_eats_m, sp1_eats_p> organism1,
            Organism<species_t, sp2_eats_m, sp2_eats_p> organism2)
  {
    // Hello!
    return std::move(organism1).ignore(std::move(organism2));
  }
};

// Plant x Plant
template<typename species_t>
struct encounter_helper<species_t, false, false, false, false>
{
  // Plants shallt not move (Matthew, probably)
  static constexpr bool possible = false;
};

// Plant x Animal
template<typename species_t, bool sp2_eats_m, bool sp2_eats_p>
  requires(sp2_eats_m || sp2_eats_p)
struct encounter_helper<species_t, false, false, sp2_eats_m, sp2_eats_p>
{
  using actual_encounter =
    encounter_helper<species_t, sp2_eats_m, sp2_eats_p, false, false>;

  static constexpr bool possible = actual_encounter::possible;

  static constexpr encounter_result_t<
    Organism<species_t, false, false>,
    Organism<species_t, sp2_eats_m, sp2_eats_p>>
  encounter(Organism<species_t, false, false> organism1,
            Organism<species_t, sp2_eats_m, sp2_eats_p> organism2)
  {
    // A plant never encounters an animal, it gets encountered
    auto [result2, result1, discard] =
      actual_encounter::encounter(std::move(organism2), std::move(organism1));
    return { std::move(result1), std::move(result2), {} };
  }
};

// Herbivore, Plant x {Carni, Omni}vore
template<typename species_t, bool sp1_eats_p, bool sp2_eats_p>
struct encounter_helper<species_t, false, sp1_eats_p, true, sp2_eats_p>
{
  using actual_encounter =
    encounter_helper<species_t, true, sp2_eats_p, false, sp1_eats_p>;

  static constexpr bool possible = actual_encounter::possible;

  static constexpr encounter_result_t<Organism<species_t, false, sp1_eats_p>,
                                      Organism<species_t, true, sp2_eats_p>>
  encounter(Organism<species_t, false, sp1_eats_p> organism1,
            Organism<species_t, true, sp2_eats_p> organism2)
  {
    // A herbivore never encounters a carnivore or omnivore, they get
    // encountered
    auto [result2, result1, discard] =
      actual_encounter::encounter(std::move(organism2), std::move(organism1));
    return { std::move(result1), std::move(result2), {} };
  }
};

// {Carni,Omni}vore x Herbivore
template<typename species_t, bool sp1_eats_p>
struct encounter_helper<species_t, true, sp1_eats_p, false, true>
{
  static constexpr bool possible = true;

  static constexpr encounter_result_t<Organism<species_t, true, sp1_eats_p>,
                                      Organism<species_t, false, true>>
  encounter(Organism<species_t, true, sp1_eats_p> organism1,
            Organism<species_t, false, true> organism2)
  {
    if (organism2.get_vitality() >= organism1.get_vitality())
      // No consumption takes place
      return std::move(organism1).ignore(std::move(organism2));
    else
      // Winner gets half vitality of loser, loser dies 💀
      return std::move(organism1).kill(std::move(organism2));
  }
};

// {Carni,Omni}vore x {Carni,Omni}vore
template<typename species_t, bool sp1_eats_p, bool sp2_eats_p>
struct encounter_helper<species_t, true, sp1_eats_p, true, sp2_eats_p>
{
  static constexpr bool possible = true;

  static constexpr encounter_result_t<Organism<species_t, true, sp1_eats_p>,
                                      Organism<species_t, true, sp2_eats_p>>
  encounter(Organism<species_t, true, sp1_eats_p> organism1,
            Organism<species_t, true, sp2_eats_p> organism2)
  {
    if (auto cmp = organism1.get_vitality() <=> organism2.get_vitality();
        cmp == std::strong_ordering::less)
      return std::move(organism1).get_killed(std::move(organism2));
    else if (cmp == std::strong_ordering::greater)
      return std::move(organism1).kill(std::move(organism2));
    else
      return { std::move(organism1).die(), std::move(organism2).die(), {} };
  }
};

// {Herbi, Omni}vore x Plant
template<typename species_t, bool sp1_eats_m>
struct encounter_helper<species_t, sp1_eats_m, true, false, false>
{
  static constexpr bool possible = true;

  static constexpr std::tuple<
    Organism<species_t, sp1_eats_m, true>,
    Organism<species_t, false, false>,
    std::optional<Organism<species_t, sp1_eats_m, true>>>
  encounter(Organism<species_t, sp1_eats_m, true> organism1,
            Organism<species_t, false, false> organism2)
  {
    return std::move(organism1).graze(std::move(organism2));
  }
};
} // namespace detail

template<typename species_t,
         bool sp1_eats_m,
         bool sp1_eats_p,
         bool sp2_eats_m,
         bool sp2_eats_p>
  requires(detail::encounter_helper<species_t,
                                    sp1_eats_m,
                                    sp1_eats_p,
                                    sp2_eats_m,
                                    sp2_eats_p>::possible)
constexpr detail::encounter_result_t<
  Organism<species_t, sp1_eats_m, sp1_eats_p>,
  Organism<species_t, sp2_eats_m, sp2_eats_p>>
encounter(Organism<species_t, sp1_eats_m, sp1_eats_p> organism1,
          Organism<species_t, sp2_eats_m, sp2_eats_p> organism2)
{
  if (organism1.is_dead() || organism2.is_dead())
    return {std::move(organism1), std::move(organism2), {}};
  if constexpr (std::is_same_v<decltype(organism1), decltype(organism2)> &&
                decltype(organism1)::is_animal)
    if (auto child = organism1.breed(organism2))
      return { std::move(organism1), std::move(organism2), std::move(child) };
  return detail::encounter_helper<species_t,
                                  sp1_eats_m,
                                  sp1_eats_p,
                                  sp2_eats_m,
                                  sp2_eats_p>::encounter(std::move(organism1),
                                                         std::move(organism2));
}

namespace detail {
template<typename species_t, bool sp1_eats_m, bool sp1_eats_p, typename... Args>
struct encounter_series_helper;

template<typename species_t, bool sp1_eats_m, bool sp1_eats_p>
struct encounter_series_helper<species_t, sp1_eats_m, sp1_eats_p>
{
  static constexpr Organism<species_t, sp1_eats_m, sp1_eats_p> encounter_series(
    Organism<species_t, sp1_eats_m, sp1_eats_p> organism)
  {
    return organism;
  }
};

template<typename species_t,
         bool sp1_eats_m,
         bool sp1_eats_p,
         typename Head,
         typename... Tail>
struct encounter_series_helper<species_t, sp1_eats_m, sp1_eats_p, Head, Tail...>
{
  static constexpr Organism<species_t, sp1_eats_m, sp1_eats_p> encounter_series(
    Organism<species_t, sp1_eats_m, sp1_eats_p> organism,
    Head&& head,
    Tail&&... tail)
  {
    return encounter_series_helper<species_t, sp1_eats_m, sp1_eats_p, Tail...>::
      encounter_series(
        std::move(std::get<0>(encounter(std::move(organism), std::move(head)))),
        std::move(tail)...);
  }
};
} // namespace detail

template<typename species_t, bool sp1_eats_m, bool sp1_eats_p, typename... Args>
constexpr Organism<species_t, sp1_eats_m, sp1_eats_p>
encounter_series(Organism<species_t, sp1_eats_m, sp1_eats_p> organism1,
                 Args... args)
{
  return detail::
    encounter_series_helper<species_t, sp1_eats_m, sp1_eats_p, Args...>::
      encounter_series(std::move(organism1), std::move(args)...);
}

#endif /* __ORGANISM_H__ */
