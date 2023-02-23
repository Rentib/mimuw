#ifndef PLAYER_H
#define PLAYER_H

#include <string>

// Klasa reprezentująca gracza. Odpowiada za przechowywanie informacji o jego
// nazwie, stanie konta, pozycji na planszy, liczbie tur czekania 
// i czy jest bankrutem.
struct Player {
  std::string const name;
  unsigned int money;
  unsigned int position;
  unsigned int waiting_turns;
  bool bankrupt;

  Player(std::string const &name)
      : name(name), money(1000), position(0), waiting_turns(0),
        bankrupt(false) {}

  void reset() noexcept {
    money = 1000;
    position = 0;
    waiting_turns = 0;
    bankrupt = false;
  }

  std::string status() const {
    if (bankrupt) {
      return "*** bankrut ***";
    }

    if (waiting_turns > 0) {
      return "*** czekanie: " + std::to_string(waiting_turns) + " ***";
    }

    return "w grze";
  }
};

#endif // PLAYER_H
