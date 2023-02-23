#ifndef GAMEBOARD_H
#define GAMEBOARD_H

#include <vector>
#include <memory>
#include "squares.h"

// Klasa representująca planszę gry. Odpowiada za przesuwanie graczy po planszy.
class GameBoard {
private:
  std::vector<std::unique_ptr<Square>> squares{};

  // Funkcje pomocznicze do obsługi przebiegu gry.
  void passSquare(Player &player) noexcept {
    player.position = (player.position + 1) % size();
    try {
      squares[player.position]->onPass(player);
    } catch (BankruptException const &e) {
      player.bankrupt = true;
      player.money = 0;
    }
  }

  void landSquare(Player &player) noexcept {
    try {
      squares[player.position]->onLand(player);
    } catch (BankruptException const &e) {
      player.bankrupt = true;
      player.money = 0;
    }
  }

public:
  GameBoard() {}
  
  void add(std::unique_ptr<Square> square) {
    squares.emplace_back(std::move(square));
  }

  size_t size() const noexcept { return squares.size(); }
  Square& get(size_t i) { return *squares[i]; }

  void movePlayer(Player &player, unsigned int distance) noexcept {
    unsigned int final_square =(player.position + distance) % size();

    for (unsigned int i = 1; i < distance; i++) {
      passSquare(player);
      if (player.bankrupt) {
        player.position = final_square;
        return;
      }
    }

    player.position = final_square;
    landSquare(player);
  }
};

#endif // GAMEBOARD_H
