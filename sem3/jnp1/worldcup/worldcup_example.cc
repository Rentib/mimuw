#include <sstream>
#include <memory>
#include <string>
#include <cassert>
#include <iostream>
#include "worldcup2022.h"

class ExampleDie : public Die {
public:
    [[nodiscard]] unsigned short roll() const override {
        static unsigned int rollNum = 0;
        unsigned short rolls[] = {1, 1, 1, 2, 1, 3};
        return rolls[rollNum++ % 6];
    }
};

class TextScoreBoard : public ScoreBoard {
    std::stringstream info;
public:
    void onRound(unsigned int roundNo) override {
        info << "=== Runda: " << roundNo << "\n";
    }

    void onTurn(std::string const &playerName, std::string const &status,
                std::string const &currentSquareName, unsigned int money) override {
        info << playerName << " [" << status << "] [" << money << "] - " << currentSquareName << "\n";
    }

    void onWin(const std::string &playerName) override {
        info << "=== Zwycięzca: " << playerName << "\n";
    }

    std::string str() {
        return info.str();
    }
};

int main() {
    std::shared_ptr<Die> die1 = std::make_shared<ExampleDie>();
    std::shared_ptr<Die> die2 = std::make_shared<ExampleDie>();
    std::shared_ptr<TextScoreBoard> scoreboard = std::make_shared<TextScoreBoard>();

    std::shared_ptr<WorldCup> worldCup2022 = std::make_shared<WorldCup2022>();
    worldCup2022->addDie(die1);
    worldCup2022->addDie(die2);
    worldCup2022->addPlayer("Lewandowski");
    worldCup2022->addPlayer("Messi");
    worldCup2022->addPlayer("Ronaldo");
    worldCup2022->setScoreBoard(scoreboard);

    worldCup2022->play(100);
    std::cout << scoreboard->str() << std::endl;
    assert(scoreboard->str() ==
           "=== Runda: 0\n"
           "Lewandowski [w grze] [840] - Dzień wolny od treningu\n"
           "Messi [w grze] [840] - Mecz z Lichtensteinem\n"
           "Ronaldo [*** czekanie: 3 ***] [620] - Żółta kartka\n"
           "=== Runda: 1\n"
           "Lewandowski [*** czekanie: 3 ***] [620] - Żółta kartka\n"
           "Messi [w grze] [540] - Mecz z Arabią Saudyjską\n"
           "Ronaldo [*** czekanie: 2 ***] [620] - Żółta kartka\n"
           "=== Runda: 2\n"
           "Lewandowski [*** czekanie: 2 ***] [620] - Żółta kartka\n"
           "Messi [w grze] [290] - Mecz z Francją\n"
           "Ronaldo [*** czekanie: 1 ***] [620] - Żółta kartka\n"
           "=== Runda: 3\n"
           "Lewandowski [*** czekanie: 1 ***] [620] - Żółta kartka\n"
           "Messi [w grze] [340] - Początek sezonu\n"
           "Ronaldo [w grze] [140] - Bukmacher\n"
           "=== Runda: 4\n"
           "Lewandowski [w grze] [665] - Mecz z Argentyną\n"
           "Messi [w grze] [180] - Dzień wolny od treningu\n"
           "Ronaldo [*** bankrut ***] [0] - Mecz z Francją\n"
           "=== Runda: 5\n"
           "Lewandowski [w grze] [315] - Początek sezonu\n"
           "Messi [*** bankrut ***] [0] - Żółta kartka\n"
           "=== Zwycięzca: Lewandowski\n");
    return 0;
}
