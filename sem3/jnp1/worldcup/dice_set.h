#ifndef DICE_SET_H
#define DICE_SET_H

#include <vector>
#include <memory>
#include "worldcup.h"

// Klasa pomocnicza reprezentująca zbiór kostek.
class DiceSet {
private:
  std::vector<std::shared_ptr<Die>> dice{};

public:
  DiceSet() {}
  void addDie(std::shared_ptr<Die> die) { dice.emplace_back(die); }
  size_t size() const noexcept { return dice.size(); }

  // Zwraca sumę oczek na wszystkich kostkach.
  unsigned int getSum() const {
    unsigned int res = 0;
    for (auto die : dice)
      res += (unsigned int)die->roll();
    return res;
  }
};

#endif // DICE_SET_H