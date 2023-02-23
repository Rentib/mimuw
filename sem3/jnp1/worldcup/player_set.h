#ifndef PLAYER_SET_H
#define PLAYER_SET_H

#include <vector>
#include <string>
#include "player.h"

// Klasa pomocnicza reprezentująca zbiór graczy.
class PlayerSet {
private:
  std::vector<Player> players{};
public:
  PlayerSet() {}

  void add(Player player) { players.emplace_back(player); }
  size_t size() const noexcept { return players.size(); }
  Player& get(size_t i) { return players[i]; }

  void reset() {
    for (auto player : players) {
      player.reset();
    }
  }

  // Zwraca imię zwycięzcy.
  std::string findWinnerName() const {
    unsigned int best_player = 0, best_score = 0;

    for (unsigned int i = 0; i < players.size(); i++) {
      if (!players[i].bankrupt) {
        if (players[i].money > best_score) {
          best_score = players[i].money;
          best_player = i;
        }
      }
    }

    return players[best_player].name;
  }
};

#endif // PLAYER_SET_H