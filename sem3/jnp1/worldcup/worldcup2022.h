#ifndef WORLDCUP2022_H
#define WORLDCUP2022_H

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "constants.h"
#include "dice_set.h"
#include "exceptions.h"
#include "gameboard.h"
#include "player.h"
#include "player_set.h"
#include "squares.h"
#include "worldcup.h"

// Główna klasa gry. Przechowuje wszystkie potrzebne elementy i odpowiada za
// przebieg rozgrywki.
class WorldCup2022 : public WorldCup {
private:
  PlayerSet players;
  std::shared_ptr<ScoreBoard> scoreboard;
  DiceSet dice;
  GameBoard game_board;
  size_t players_count = 0;

  void startGame() {
    players.reset();
    game_board = GameBoard{};
    players_count = players.size();

    game_board.add(std::make_unique<StartSquare>("Początek sezonu"));
    game_board.add(std::make_unique<FriendlyMatchSquare>("Mecz z San Marino", 160));
    game_board.add(std::make_unique<DayOffSquare>("Dzień wolny od treningu"));
    game_board.add(std::make_unique<FriendlyMatchSquare>("Mecz z Lichtensteinem", 220));
    game_board.add(std::make_unique<YellowCardSquare>("Żółta kartka", 3));
    game_board.add(std::make_unique<PointsMatchSquare>("Mecz z Meksykiem", 300));
    game_board.add(std::make_unique<PointsMatchSquare>("Mecz z Arabią Saudyjską", 280));
    game_board.add(std::make_unique<BookmakerSquare>("Bukmacher", 100, 100));
    game_board.add(std::make_unique<PointsMatchSquare>("Mecz z Argentyną", 250));
    game_board.add(std::make_unique<GoalSquare>("Gol", 120));
    game_board.add(std::make_unique<FinalMatchSquare>("Mecz z Francją", 400));
    game_board.add(std::make_unique<PenaltySquare>("Rzut karny", 180));
  }

  void playerTurn(Player &player) {
    if (player.bankrupt) {
      return;
    }

    if (player.waiting_turns > 0) {
      player.waiting_turns--;
      if (player.waiting_turns > 0)
        return;
    }

    unsigned int dice_result = dice.getSum();
    game_board.movePlayer(player, dice_result);
  }

  void singleRound() {
    for (unsigned int player = 0; player < players.size(); player++) {
      Player &current = players.get(player);
      if (current.bankrupt) {
        continue;
      }

      playerTurn(current);
      if (current.bankrupt) {
        players_count--;
      }

      scoreboard->onTurn(current.name, current.status(),
                          game_board.get(current.position).getName(),
                          current.money);

      if (players_count == 1) {
        scoreboard->onWin(players.findWinnerName());
        return;
      }
    }
  }

  void checkDice() {
    if (dice.size() > MAX_DICE) {
      throw TooManyDiceException();
    } else if (dice.size() < MIN_DICE) {
      throw TooFewDiceException();
    }
  }

  void checkPlayers() {
    if (players.size() > MAX_PLAYERS) {
      throw TooManyPlayersException();
    } else if (players.size() < MIN_PLAYERS) {
      throw TooFewPlayersException();
    }
  }

public:
  explicit WorldCup2022() : game_board(GameBoard{}) {}

  // Jeżeli argumentem jest pusty wskaźnik, to nie wykonuje żadnej operacji
  // (ale nie ma błędu).
  void addDie(std::shared_ptr<Die> die) override {
    if (die != nullptr) {
      dice.addDie(die);
    }
  }

  // Dodaje nowego gracza o podanej nazwie.
  void addPlayer(std::string const &name) override {
    players.add(Player(name));
  }

  // Konfiguruje tablicę wyników. Domyślnie jest skonfigurowana tablica
  // wyników, która nic nie robi.
  void setScoreBoard(std::shared_ptr<ScoreBoard> scoreboard) override {
    if (scoreboard != nullptr) {
      this->scoreboard = scoreboard;
    } else {
      throw std::invalid_argument("scoreboard is null");
    }
  }

  void play(unsigned int rounds) override {
    checkDice();
    checkPlayers();
    startGame();

    for (unsigned int round = 0; round < rounds; round++) {
      scoreboard->onRound(round);
      singleRound();

      if (players_count == 1) {
        return;
      }
    }
    
    scoreboard->onWin(players.findWinnerName());
  }
};

#endif // WORLDCUP2022_H
