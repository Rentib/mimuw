#ifndef EXCEPTIONS_H
#define EXCEPTIONS_H

#include <stdexcept>

class TooManyDiceException : public std::exception {
public:
  const char *what() const noexcept override { return "Too many dice"; }
};

class TooFewDiceException : public std::exception {
public:
  const char *what() const noexcept override { return "Too few dice"; }
};

class TooManyPlayersException : public std::exception {
public:
  const char *what() const noexcept override { return "Too many players"; }
};

class TooFewPlayersException : public std::exception {
public:
  const char *what() const noexcept override { return "Too few players"; }
};

  // Ten wyjątek służy do sygnalizacji bankructwa gracza.
class BankruptException : public std::exception {
public:
  const char *what() const noexcept override { return "Bankrupt"; }
};

#endif // EXCEPTIONS_H
