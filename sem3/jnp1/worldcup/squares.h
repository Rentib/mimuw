#ifndef SQUARES_H
#define SQUARES_H

#include <string>
#include "player.h"
#include "exceptions.h"

// Klasa bazowa pola. Każde pole musi mieć nazwę i implementować funkcje
// onLand i onPass. Pola odpowiadają za zmianę stanu gracza po wejściu
// lub przejściu przez nie.
class Square {
protected:
    std::string square_name;

public:
    virtual ~Square() = default;
    explicit Square(std::string const &square_name)
        : square_name(square_name) {}
    std::string const &getName() const noexcept { return square_name; }

    virtual void onLand(Player &player) = 0;
    virtual void onPass(Player &player) = 0;
};

class StartSquare : public Square {
public:
  explicit StartSquare(std::string const &name) : Square(name) {}
  void onLand(Player &player) noexcept override { player.money += 50; }
  void onPass(Player &player) noexcept override { player.money += 50; }
};

// Pola dające / zabierające pewną ustalona kwotę mogą 
// dziedziczyć po tej klasie.
class BonusSquare : public Square {
protected:
  unsigned int bonus;

public:
  virtual ~BonusSquare() = default;
  explicit BonusSquare(std::string const &name, unsigned int bonus)
      : Square(name), bonus(bonus) {}

  void onPass([[maybe_unused]] Player &player) noexcept override{};
};

class GoalSquare : public BonusSquare {
public:
  explicit GoalSquare(std::string const &name, unsigned int bonus)
      : BonusSquare(name, bonus) {}
  void onLand(Player &player) noexcept override { player.money += bonus; }
};

class PenaltySquare : public BonusSquare {
public:
  explicit PenaltySquare(std::string const &name, unsigned int bonus)
      : BonusSquare(name, bonus) {}
  void onLand(Player &player) override {
    if (player.money < bonus) {
      throw BankruptException();
    }

    player.money -= bonus;
  }
};

class BookmakerSquare : public Square {
private:
  unsigned int count;
  unsigned int lose;
  unsigned int win;

public:
  explicit BookmakerSquare(std::string const &name, unsigned int lose,
                            unsigned int win)
      : Square(name), count(0), lose(lose), win(win) {}

  void onLand(Player &player) override {
    count = (count + 1) % 3;
    if (count == 1) {
      player.money += win;
    } else {
      if (player.money < lose) {
        throw BankruptException();
      }
      player.money -= lose;
    }
  }

  void onPass([[maybe_unused]] Player &player) noexcept override {}
};

class YellowCardSquare : public Square {
private:
  unsigned int waiting_time;

public:
  explicit YellowCardSquare(std::string const &name,
                            unsigned int waiting_time)
      : Square(name), waiting_time(waiting_time) {}

  void onLand(Player &player) noexcept override {
    player.waiting_turns += waiting_time;
  }

  void onPass([[maybe_unused]] Player &player) noexcept override {}
};

// Klasa bazowa pola meczu. Różne typy meczów są reprezentowane
// przez odpowiednie klasy pochodne.
class MatchSquare : public Square {
protected:
  unsigned int price;
  unsigned int collected;

public:
  virtual ~MatchSquare() = default;
  explicit MatchSquare(std::string const &name, unsigned int price)
      : Square(name), price(price), collected(0) {}

  void onPass(Player &player) override {
    collected += std::min(player.money, price);

    if (player.money < price) {
      throw BankruptException();
    }

    player.money -= price;
  }
};

class FriendlyMatchSquare : public MatchSquare {
public:
  explicit FriendlyMatchSquare(std::string const &name, unsigned int price)
      : MatchSquare(name, price) {}

  void onLand(Player &player) noexcept override {
    player.money += collected * 1;
    collected = 0;
  }
};

class PointsMatchSquare : public MatchSquare {
public:
  explicit PointsMatchSquare(std::string const &name, unsigned int price)
      : MatchSquare(name, price) {}

  void onLand(Player &player) noexcept override {
    player.money += collected * 2 + collected / 2;
    collected = 0;
  }
};

class FinalMatchSquare : public MatchSquare {
public:
  explicit FinalMatchSquare(std::string const &name, unsigned int price)
      : MatchSquare(name, price) {}

  void onLand(Player &player) noexcept override {
    player.money += collected * 4;
    collected = 0;
  }
};

class DayOffSquare : public Square {
public:
  explicit DayOffSquare(std::string const &name) : Square(name) {}

  void onLand([[maybe_unused]] Player &player) noexcept override {}
  void onPass([[maybe_unused]] Player &player) noexcept override {}
};


#endif // SQUARES_H
