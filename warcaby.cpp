// Kompilacja (Linux/WSL):
// g++ warcaby.cpp -o warcaby -lsfml-graphics -lsfml-window -lsfml-system -pthread

#include <SFML/Graphics.hpp>
#include <iostream>
#include <vector>
#include <limits>
#include <cmath>
#include <chrono>
#include <string>
#include <cstdio>
#include <thread>
#include <atomic>
#include <mutex>
#include <algorithm>
#include <unordered_map>
#include <random>

using namespace std;
using Clock = std::chrono::steady_clock;

// Reprezentacja pionków:
//  0  – puste
//  1  – biały pion
//  2  – biała damka
// -1  – czarny pion
// -2  – czarna damka

struct Move {
    vector<pair<int,int>> path;     // kolejne pola (x,y)
    vector<pair<int,int>> captured; // zbite pionki (x,y)
};

class CheckersGame {
public:
    static const int SIZE = 8;

    int board[SIZE][SIZE];
    bool whiteTurn = true;
    bool gameOver = false;
    int  winner   = 0;  // 1 białe, -1 czarne, 0 remis/brak

    CheckersGame() {
        resetBoard();
    }

    void resetBoard() {
        gameOver  = false;
        winner    = 0;
        whiteTurn = true;

        for (int y = 0; y < SIZE; ++y)
            for (int x = 0; x < SIZE; ++x)
                board[y][x] = 0;

        // Czarne na górze (pola ciemne)
        for (int y = 0; y < 3; ++y) {
            for (int x = 0; x < SIZE; ++x) {
                if ((x + y) % 2 == 1) board[y][x] = -1;
            }
        }
        // Białe na dole
        for (int y = SIZE - 3; y < SIZE; ++y) {
            for (int x = 0; x < SIZE; ++x) {
                if ((x + y) % 2 == 1) board[y][x] = 1;
            }
        }
    }

    bool inside(int x, int y) const {
        return x >= 0 && x < SIZE && y >= 0 && y < SIZE;
    }

    bool isWhite(int v) const { return v == 1 || v == 2; }
    bool isBlack(int v) const { return v == -1 || v == -2; }
    bool isKing (int v) const { return v == 2 || v == -2; }

    // --------------------------
    // GENEROWANIE RUCHÓW
    // --------------------------
    vector<Move> getAllMoves(bool forWhite) {
        vector<Move> captureMoves;
        vector<Move> normalMoves;

        for (int y = 0; y < SIZE; ++y) {
            for (int x = 0; x < SIZE; ++x) {
                int p = board[y][x];
                if (p == 0) continue;
                if (forWhite && !isWhite(p)) continue;
                if (!forWhite && !isBlack(p)) continue;

                vector<Move> localCaptures;
                generateCapturesFrom(x, y, localCaptures);

                if (!localCaptures.empty()) {
                    // Zapisujemy wszystkie bicia – brak zasady bicia większości
                    captureMoves.insert(captureMoves.end(),
                                        localCaptures.begin(),
                                        localCaptures.end());
                } else {
                    vector<Move> localMoves;
                    generateNormalMovesFrom(x, y, localMoves);
                    normalMoves.insert(normalMoves.end(),
                                       localMoves.begin(),
                                       localMoves.end());
                }
            }
        }

        // Bicie obowiązkowe – jeśli jest jakiekolwiek bicie, zwracamy tylko bicia
        if (!captureMoves.empty()) return captureMoves;
        return normalMoves;
    }

    // zwykłe ruchy (bez bicia)
    void generateNormalMovesFrom(int x, int y, vector<Move>& moves) {
        int p = board[y][x];
        if (p == 0) return;

        if (isKing(p)) {
            // Król: krótkie ruchy o jedno pole w każdym kierunku
            const int dirs[4][2] = { {1,1}, {-1,1}, {1,-1}, {-1,-1} };
            for (auto &d : dirs) {
                int nx = x + d[0];
                int ny = y + d[1];
                if (inside(nx, ny) && board[ny][nx] == 0) {
                    Move m;
                    m.path.push_back({x, y});
                    m.path.push_back({nx, ny});
                    moves.push_back(m);
                }
            }
        } else {
            // Pionek: ruch tylko do przodu (dla białych w górę, dla czarnych w dół)
            int dir = (p > 0) ? -1 : 1;
            for (int dx : {-1, 1}) {
                int nx = x + dx;
                int ny = y + dir;
                if (inside(nx, ny) && board[ny][nx] == 0) {
                    Move m;
                    m.path.push_back({x, y});
                    m.path.push_back({nx, ny});
                    moves.push_back(m);
                }
            }
        }
    }

    // bicia (może być wielokrotne)
    void generateCapturesFrom(int x, int y, vector<Move>& result) {
        int p = board[y][x];
        if (p == 0) return;

        Move base;
        base.path.push_back({x, y});
        dfsCaptures(x, y, p, base, result);
    }

    // Zgodnie z zasadami:
    // - pionek może bić zarówno do przodu, jak i do tyłu,
    // - król też, ale zawsze „skok” o 2 pola.
    void dfsCaptures(int x, int y, int p, Move current, vector<Move>& result) {
        bool any = false;

        const int dirs[4][2] = { {1,1}, {-1,1}, {1,-1}, {-1,-1} };

        for (auto &d : dirs) {
            int mx = x + d[0];
            int my = y + d[1];
            int lx = x + 2*d[0];
            int ly = y + 2*d[1];

            if (!inside(mx, my) || !inside(lx, ly)) continue;

            int mid = board[my][mx];
            if (mid == 0) continue;

            bool enemy =
                (p > 0 && isBlack(mid)) ||
                (p < 0 && isWhite(mid));

            if (!enemy) continue;
            if (board[ly][lx] != 0) continue;

            // wykonujemy skok
            int backup[SIZE][SIZE];
            copyBoard(backup, board);

            board[y][x]   = 0;
            board[my][mx] = 0;
            board[ly][lx] = p;

            Move next = current;
            next.path.push_back({lx, ly});
            next.captured.push_back({mx, my});

            int newP = p;
            // awans na damkę – po dojściu do końca
            if (p == 1 && ly == 0)       newP = 2;
            if (p == -1 && ly == SIZE-1) newP = -2;
            board[ly][lx] = newP;

            dfsCaptures(lx, ly, newP, next, result);

            copyBoard(board, backup);
            any = true;
        }

        // jeśli nie ma dalszych bić, a cokolwiek zbiliśmy – to gotowy ruch
        if (!any && !current.captured.empty()) {
            result.push_back(current);
        }
    }

    void copyBoard(int dst[SIZE][SIZE], int src[SIZE][SIZE]) {
        for (int y = 0; y < SIZE; ++y)
            for (int x = 0; x < SIZE; ++x)
                dst[y][x] = src[y][x];
    }

    // --------------------------
    // Wykonanie ruchu
    // --------------------------
    void applyMove(const Move& m) {
        if (m.path.size() < 2) return;

        int fromX = m.path.front().first;
        int fromY = m.path.front().second;
        int toX   = m.path.back().first;
        int toY   = m.path.back().second;

        int p = board[fromY][fromX];
        board[fromY][fromX] = 0;

        for (auto &c : m.captured) {
            board[c.second][c.first] = 0;
        }

        board[toY][toX] = p;

        // awans
        if (p == 1 && toY == 0)       board[toY][toX] = 2;
        if (p == -1 && toY == SIZE-1) board[toY][toX] = -2;

        checkGameOver();
    }

    // do minimax (bez checkGameOver)
    void applyMoveNoCheck(const Move& m) {
        if (m.path.size() < 2) return;

        int fromX = m.path.front().first;
        int fromY = m.path.front().second;
        int toX   = m.path.back().first;
        int toY   = m.path.back().second;

        int p = board[fromY][fromX];
        board[fromY][fromX] = 0;

        for (auto &c : m.captured) {
            board[c.second][c.first] = 0;
        }

        board[toY][toX] = p;

        if (p == 1 && toY == 0)       board[toY][toX] = 2;
        if (p == -1 && toY == SIZE-1) board[toY][toX] = -2;

        gameOver = false;
        winner   = 0;
    }

    void checkGameOver() {
        int wc = 0, bc = 0;
        for (int y = 0; y < SIZE; ++y)
            for (int x = 0; x < SIZE; ++x) {
                if (isWhite(board[y][x])) wc++;
                if (isBlack(board[y][x])) bc++;
            }

        vector<Move> wMoves = getAllMoves(true);
        vector<Move> bMoves = getAllMoves(false);
        bool wNoMove = wMoves.empty();
        bool bNoMove = bMoves.empty();

        if (wc == 0 && bc > 0) { gameOver = true; winner = -1; return; }
        if (bc == 0 && wc > 0) { gameOver = true; winner =  1; return; }
        if (wc == 0 && bc == 0){ gameOver = true; winner =  0; return; }

        if (wNoMove || bNoMove) {
            int res = 0;
            if (wc > bc)      res = 1;
            else if (bc > wc) res = -1;
            else              res = 0;
            gameOver = true;
            winner   = res;
        }
    }

    // -----------------------
    // Ewaluacja (dla białych)
    // -----------------------
    int evaluate() {
        const int MAN_VAL  = 100;
        const int KING_VAL = 250;

        int score = 0;
        int wMen = 0, bMen = 0, wKings = 0, bKings = 0;

        static const int PSQT[SIZE][SIZE] = {
            {  0,  1,  2,  3,  3,  2,  1,  0 },
            {  1,  2,  3,  4,  4,  3,  2,  1 },
            {  2,  3,  4,  5,  5,  4,  3,  2 },
            {  3,  4,  5,  6,  6,  5,  4,  3 },
            {  3,  4,  5,  6,  6,  5,  4,  3 },
            {  2,  3,  4,  5,  5,  4,  3,  2 },
            {  1,  2,  3,  4,  4,  3,  2,  1 },
            {  0,  1,  2,  3,  3,  2,  1,  0 }
        };

        for (int y = 0; y < SIZE; ++y) {
            for (int x = 0; x < SIZE; ++x) {
                int p = board[y][x];
                if (p == 0) continue;
                bool white = isWhite(p);
                bool king  = isKing(p);

                // materiał
                if (p == 1)      { score += MAN_VAL;  wMen++;   }
                else if (p == 2) { score += KING_VAL; wKings++; }
                else if (p == -1){ score -= MAN_VAL;  bMen++;   }
                else if (p == -2){ score -= KING_VAL; bKings++; }

                // zaawansowanie pionków – tylko zwykłe piony
                if (!king) {
                    int adv = white ? (7 - y) : y;
                    int advBonus = adv * 6;
                    score += white ? advBonus : -advBonus;
                }

                // centralizacja
                int ps = PSQT[y][x];
                int centerBonus = ps * (king ? 4 : 2);
                score += white ? centerBonus : -centerBonus;
            }
        }

        int totalPieces = wMen + bMen + wKings + bKings;

        // mobilność + informacja o możliwych bicich
        vector<Move> wMoves = getAllMoves(true);
        vector<Move> bMoves = getAllMoves(false);
        int wm = (int)wMoves.size();
        int bm = (int)bMoves.size();

        if (wm == 0) score -= 800;
        if (bm == 0) score += 800;
        score += (wm - bm) * 4;

        bool wCanCapture = !wMoves.empty() && !wMoves[0].captured.empty();
        bool bCanCapture = !bMoves.empty() && !bMoves[0].captured.empty();
        if (wCanCapture) score += 30;
        if (bCanCapture) score -= 30;

        // końcówki
        if (totalPieces <= 8) {
            score += (wKings - bKings) * 40;
            score += (wMen   - bMen)   * 10;
        }

        return score;
    }

    int minimax(int depth, int alpha, int beta, bool whiteToMove,
                Clock::time_point startTime, double timeLimitMs,
                bool inQuiescence = false);

    void restoreState(int savedBoard[SIZE][SIZE],
                      bool savedTurn,
                      bool savedGameOver,
                      int savedWinner) {
        copyBoard(board, savedBoard);
        whiteTurn = savedTurn;
        gameOver  = savedGameOver;
        winner    = savedWinner;
    }
};

// ---------------------- Zobrist + tablica transpozycji -----------------

static uint64_t Z_PIECE[5][CheckersGame::SIZE][CheckersGame::SIZE];
static uint64_t Z_SIDE;
static bool Z_INITIALIZED = false;

static std::mt19937_64 Z_RNG(20251116);

uint64_t rand64() {
    std::uniform_int_distribution<uint64_t> dist;
    return dist(Z_RNG);
}

void initZobrist() {
    if (Z_INITIALIZED) return;
    for (int t = 0; t < 5; ++t)
        for (int y = 0; y < CheckersGame::SIZE; ++y)
            for (int x = 0; x < CheckersGame::SIZE; ++x)
                Z_PIECE[t][y][x] = rand64();
    Z_SIDE = rand64();
    Z_INITIALIZED = true;
}

inline int pieceIndexFromValue(int v) {
    switch (v) {
        case  1: return 1;
        case  2: return 2;
        case -1: return 3;
        case -2: return 4;
        default: return 0;
    }
}

uint64_t computeHash(const CheckersGame& g, bool whiteToMove) {
    uint64_t h = 0;
    for (int y = 0; y < CheckersGame::SIZE; ++y) {
        for (int x = 0; x < CheckersGame::SIZE; ++x) {
            int v = g.board[y][x];
            int idx = pieceIndexFromValue(v);
            if (idx != 0)
                h ^= Z_PIECE[idx][y][x];
        }
    }
    if (whiteToMove) h ^= Z_SIDE;
    return h;
}

struct TTEntry {
    int depth;
    int value;
};

static std::unordered_map<uint64_t, TTEntry> TT;
static std::mutex TTMutex;

// ---------------- MINIMAX + QUIESCENCE + TT ----------------

int CheckersGame::minimax(int depth, int alpha, int beta, bool whiteToMove,
                          Clock::time_point startTime, double timeLimitMs,
                          bool inQuiescence)
{
    // 1. Limit czasu – jak się skończy, zwracamy statyczną ewaluację
    if (timeLimitMs > 0) {
        auto now = Clock::now();
        double elapsed =
            std::chrono::duration_cast<std::chrono::milliseconds>(now - startTime).count();
        if (elapsed > timeLimitMs) {
            return evaluate();
        }
    }

    // 2. Generujemy wszystkie ruchy
    vector<Move> moves = getAllMoves(whiteToMove);
    bool hasCapture = !moves.empty() && !moves[0].captured.empty();

    // Jeśli jesteśmy w trybie quiescence – zostawiamy tylko ruchy z biciem
    if (inQuiescence) {
        vector<Move> capMoves;
        capMoves.reserve(moves.size());
        for (const auto& m : moves) {
            if (!m.captured.empty())
                capMoves.push_back(m);
        }
        moves.swap(capMoves);
        hasCapture = !moves.empty();
    }

    // 3. Warunek zatrzymania
    if (depth == 0) {
        // Nie w quiescence, ale są bicia – wejdź w quiescence tylko z bicami
        if (!inQuiescence && hasCapture) {
            inQuiescence = true;

            vector<Move> capMoves;
            capMoves.reserve(moves.size());
            for (const auto& m : moves) {
                if (!m.captured.empty())
                    capMoves.push_back(m);
            }
            moves.swap(capMoves);
            hasCapture = !moves.empty();

            if (!hasCapture) {
                return evaluate();
            }
        } else {
            return evaluate();
        }
    }

    // 4. Brak ruchów – przegrana/wygrana od strony bez ruchu
    if (moves.empty()) {
        return whiteToMove ? -1000000 : 1000000;
    }

    // 5. Transposition table – tylko poza quiescence
    uint64_t hash = 0;
    if (!inQuiescence) {
        hash = computeHash(*this, whiteToMove);
        {
            std::lock_guard<std::mutex> lock(TTMutex);
            auto it = TT.find(hash);
            if (it != TT.end() && it->second.depth >= depth) {
                return it->second.value;
            }
        }
    }

    // 6. Backup stanu
    int  backup[SIZE][SIZE];
    copyBoard(backup, board);
    bool backupTurn     = whiteTurn;
    bool backupGameOver = gameOver;
    int  backupWinner   = winner;

    int result = 0;

    if (whiteToMove) {
        int maxEval = std::numeric_limits<int>::min();

        for (const auto& m : moves) {
            applyMoveNoCheck(m);
            whiteTurn = !whiteTurn;

            int nextDepth = (depth > 0) ? depth - 1 : 0;

            int eval = minimax(nextDepth,
                               alpha, beta,
                               false,
                               startTime, timeLimitMs,
                               inQuiescence);

            restoreState(backup, backupTurn, backupGameOver, backupWinner);

            if (eval > maxEval) maxEval = eval;
            if (eval > alpha)   alpha   = eval;
            if (beta <= alpha) break;
        }
        result = maxEval;
    } else {
        int minEval = std::numeric_limits<int>::max();

        for (const auto& m : moves) {
            applyMoveNoCheck(m);
            whiteTurn = !whiteTurn;

            int nextDepth = (depth > 0) ? depth - 1 : 0;

            int eval = minimax(nextDepth,
                               alpha, beta,
                               true,
                               startTime, timeLimitMs,
                               inQuiescence);

            restoreState(backup, backupTurn, backupGameOver, backupWinner);

            if (eval < minEval) minEval = eval;
            if (eval < beta)    beta    = eval;
            if (beta <= alpha) break;
        }
        result = minEval;
    }

    // 7. Zapis do TT – tylko główne drzewo
    if (!inQuiescence) {
        std::lock_guard<std::mutex> lock(TTMutex);
        auto it = TT.find(hash);
        if (it == TT.end() || it->second.depth < depth) {
            TT[hash] = TTEntry{ depth, result };
        }
    }

    return result;
}

// ---------------- POMOCNICZE TEKSTOWE ----------------

string coordToStr(int x, int y) {
    char col = 'A' + x;
    char row = '1' + (7 - y);
    string s;
    s += col;
    s += row;
    return s;
}

string moveToStr(const Move& m, bool white) {
    string s = white ? "B: " : "C: ";
    if (m.path.empty()) return s + "?";
    bool capture = !m.captured.empty();
    char sep = capture ? 'x' : '-';
    for (size_t i = 0; i < m.path.size(); ++i) {
        if (i > 0) s += sep;
        s += coordToStr(m.path[i].first, m.path[i].second);
    }
    return s;
}

string formatTime(float seconds) {
    if (seconds < 0) seconds = 0;
    int total = static_cast<int>(seconds + 0.5f);
    int m = total / 60;
    int s = total % 60;
    char buf[16];
    snprintf(buf, sizeof(buf), "%02d:%02d", m, s);
    return string(buf);
}

struct GameSnapshot {
    int  board[CheckersGame::SIZE][CheckersGame::SIZE];
    bool whiteTurn;
    bool gameOver;
    int  winner;
    Move lastMove;
    vector<string> moveHistory;
    float whiteTime;
    float blackTime;
};

struct ColorOption {
    sf::Color color;
    const char* name;
};

static const ColorOption COLOR_OPTIONS[] = {
    { sf::Color::White,        "Bialy" },
    { sf::Color::Black,        "Czarny" },
    { sf::Color::Red,          "Czerwony" },
    { sf::Color::Blue,         "Niebieski" },
    { sf::Color::Green,        "Zielony" },
    { sf::Color::Yellow,       "Zolty" },
    { sf::Color::Magenta,      "Magenta" },
    { sf::Color::Cyan,         "Cyan" },
    { sf::Color(255,165,0),    "Pomaranczowy" }
};
static const int COLOR_COUNT = sizeof(COLOR_OPTIONS)/sizeof(ColorOption);

// ---------------- RÓWNOLEGŁE SZUKANIE ----------------

struct ThreadResult {
    int  bestEval;
    Move bestMove;
};

void aiWorkerRange(CheckersGame baseGame,
                   const vector<Move>& moves,
                   int startIdx, int endIdx,
                   int depth, double timeLimitMs,
                   bool aiIsWhite,
                   ThreadResult& out)
{
    int localBestEval = aiIsWhite
                        ? std::numeric_limits<int>::min()
                        : std::numeric_limits<int>::max();
    Move localBestMove;

    int  backup[CheckersGame::SIZE][CheckersGame::SIZE];
    bool backupTurn, backupGameOver;
    int  backupWinner;

    Clock::time_point startTime = Clock::now();

    for (int i = startIdx; i < endIdx; ++i) {
        CheckersGame g = baseGame;

        g.copyBoard(backup, g.board);
        backupTurn     = g.whiteTurn;
        backupGameOver = g.gameOver;
        backupWinner   = g.winner;

        g.applyMoveNoCheck(moves[i]);
        g.whiteTurn = !g.whiteTurn;

        bool whiteToMoveNext = !aiIsWhite;

        int eval = g.minimax(depth - 1,
                             std::numeric_limits<int>::min(),
                             std::numeric_limits<int>::max(),
                             whiteToMoveNext,
                             startTime,
                             timeLimitMs,
                             false);

        g.restoreState(backup, backupTurn, backupGameOver, backupWinner);

        if (aiIsWhite) {
            if (eval > localBestEval) {
                localBestEval = eval;
                localBestMove = moves[i];
            }
        } else {
            if (eval < localBestEval) {
                localBestEval = eval;
                localBestMove = moves[i];
            }
        }
    }

    out.bestEval = localBestEval;
    out.bestMove = localBestMove;
}

Move parallelSearch(const CheckersGame& game,
                    bool forWhite,
                    int depth, double timeLimitMs)
{
    CheckersGame base = game;
    vector<Move> moves = base.getAllMoves(forWhite);
    Move best;
    if (moves.empty()) return best;

    auto moveScore = [&](const Move& m) {
        int s = 0;
        s += (int)m.captured.size() * 200;
        if (!m.path.empty()) {
            auto dest = m.path.back();
            int x = dest.first;
            int y = dest.second;
            int dx = std::min(x, 7 - x);
            int dy = std::min(y, 7 - y);
            int distEdge = std::min(dx, dy);
            int centerBonus = 3 - distEdge;
            if (centerBonus < 0) centerBonus = 0;
            s += centerBonus * 10;
        }
        return s;
    };

    std::sort(moves.begin(), moves.end(),
              [&](const Move& a, const Move& b) {
                  return moveScore(a) > moveScore(b);
              });

    unsigned int hw = std::thread::hardware_concurrency();
    if (hw == 0) hw = 2;
    unsigned int threadsCount = std::min(hw, (unsigned int)moves.size());

    vector<std::thread>   threads;
    vector<ThreadResult>  results(threadsCount);

    int chunk = (int)moves.size() / threadsCount;
    int rem   = (int)moves.size() % threadsCount;
    int start = 0;

    bool aiIsWhite = forWhite;

    for (unsigned int t = 0; t < threadsCount; ++t) {
        int count = chunk + (t < (unsigned int)rem ? 1 : 0);
        int end   = start + count;
        if (start >= end) {
            results[t].bestEval = aiIsWhite
                                  ? std::numeric_limits<int>::min()
                                  : std::numeric_limits<int>::max();
            continue;
        }
        results[t].bestEval = aiIsWhite
                              ? std::numeric_limits<int>::min()
                              : std::numeric_limits<int>::max();

        threads.emplace_back(aiWorkerRange,
                             base,
                             std::cref(moves),
                             start, end,
                             depth, timeLimitMs,
                             aiIsWhite,
                             std::ref(results[t]));
        start = end;
    }

    for (auto& th : threads) th.join();

    if (aiIsWhite) {
        int globalBestEval = std::numeric_limits<int>::min();
        for (auto &r : results) {
            if (r.bestEval > globalBestEval) {
                globalBestEval = r.bestEval;
                best = r.bestMove;
            }
        }
    } else {
        int globalBestEval = std::numeric_limits<int>::max();
        for (auto &r : results) {
            if (r.bestEval < globalBestEval) {
                globalBestEval = r.bestEval;
                best = r.bestMove;
            }
        }
    }
    return best;
}

// --------------- ITERATIVE DEEPENING ----------------

const int   MAX_SEARCH_DEPTH    = 24;
const double AI_TIME_LIMIT_MS   = 10000;  // ok. 10 s
const double HINT_TIME_LIMIT_MS = 10000;  // ok. 10 s

Move searchBestMoveStrong(const CheckersGame& game,
                          bool forWhite,
                          double totalTimeMs)
{
    CheckersGame tmp = game;
    vector<Move> moves = tmp.getAllMoves(forWhite);
    if (moves.empty()) return Move{};
    if (moves.size() == 1) return moves[0];

    Move best;
    bool haveAny = false;
    auto start = Clock::now();

    for (int depth = 2; depth <= MAX_SEARCH_DEPTH; ++depth) {
        auto now = Clock::now();
        double elapsed =
            std::chrono::duration_cast<std::chrono::milliseconds>(now - start).count();
        double remaining = totalTimeMs - elapsed;
        if (remaining < 200.0) break;

        Move candidate = parallelSearch(game, forWhite, depth, remaining);
        if (candidate.path.size() >= 2) {
            best = candidate;
            haveAny = true;
        }
    }

    if (!haveAny) {
        best = parallelSearch(game, forWhite, 2, totalTimeMs);
    }
    return best;
}

// ---------------- GŁÓWNA PĘTLA SFML ----------------

enum class GameState {
    MENU,
    PLAY
};

int main() {
    initZobrist();

    CheckersGame game;

    const int TILE_SIZE   = 80;
    const int BOARD_PX    = CheckersGame::SIZE * TILE_SIZE;
    const int HEADER_H    = 80;
    const int LEFT_PANEL  = 200;
    const int RIGHT_PANEL = 260;
    const int WINDOW_W    = LEFT_PANEL + BOARD_PX + RIGHT_PANEL;
    const int WINDOW_H    = HEADER_H + BOARD_PX;

    sf::RenderWindow window(
        sf::VideoMode(WINDOW_W, WINDOW_H),
        "Warcaby - silnik wielowatkowy"
    );
    window.setFramerateLimit(60);

    sf::Font font;
    if (!font.loadFromFile("arial.ttf")) {
        std::cout << "UWAGA: brak arial.ttf – napisy nie beda widoczne.\n";
    }

    GameState state         = GameState::MENU;
    bool      twoPlayerMode = false;
    bool      humanIsWhite  = true;
    bool      aiIsWhite     = false;

    bool hintsEnabled = true;

    int playerColorIdx   = 0;
    int opponentColorIdx = 1;

    int selectedX = -1, selectedY = -1;
    vector<Move> movesFromSelected;
    Move lastMove;

    bool  isAnimating       = false;
    Move  animMove;
    size_t animSegmentIndex = 0;
    float animTime          = 0.f;
    const float SEG_DURATION = 0.2f;
    int   animPieceValue    = 0;

    vector<string> moveHistory;

    vector<GameSnapshot> snapshots;

    float    whiteTotalTime = 0.f;
    float    blackTotalTime = 0.f;
    sf::Clock turnClock;

    auto pushSnapshot = [&]() {
        GameSnapshot snap;
        game.copyBoard(snap.board, game.board);
        snap.whiteTurn   = game.whiteTurn;
        snap.gameOver    = game.gameOver;
        snap.winner      = game.winner;
        snap.lastMove    = lastMove;
        snap.moveHistory = moveHistory;
        snap.whiteTime   = whiteTotalTime;
        snap.blackTime   = blackTotalTime;
        snapshots.push_back(snap);
    };

    auto restoreFromSnapshot = [&]() {
        if (snapshots.empty()) return;
        GameSnapshot snap = snapshots.back();
        snapshots.pop_back();
        game.copyBoard(game.board, snap.board);
        game.whiteTurn   = snap.whiteTurn;
        game.gameOver    = snap.gameOver;
        game.winner      = snap.winner;
        lastMove         = snap.lastMove;
        moveHistory      = snap.moveHistory;
        whiteTotalTime   = snap.whiteTime;
        blackTotalTime   = snap.blackTime;
        selectedX = selectedY = -1;
        movesFromSelected.clear();
        isAnimating = false;
        animMove.path.clear();
        animMove.captured.clear();
        animSegmentIndex = 0;
        animTime = 0.f;
        turnClock.restart();
    };

    auto resetGame = [&]() {
        game.resetBoard();
        selectedX = selectedY = -1;
        movesFromSelected.clear();
        lastMove.path.clear();
        lastMove.captured.clear();
        isAnimating = false;
        animMove.path.clear();
        animMove.captured.clear();
        animSegmentIndex = 0;
        animTime = 0.f;
        moveHistory.clear();
        snapshots.clear();
        whiteTotalTime = 0.f;
        blackTotalTime = 0.f;
        game.gameOver  = false;
        game.winner    = 0;
        game.whiteTurn = true;
        {
            std::lock_guard<std::mutex> lock(TTMutex);
            TT.clear();
        }
        turnClock.restart();
        pushSnapshot();
    };

    resetGame();

    std::atomic<bool> aiThinking(false);
    std::atomic<bool> aiMoveReady(false);
    Move aiMoveResult;
    std::mutex aiMutex;

    auto startAIThinking = [&](bool aiIsWhiteLocal) {
        if (aiThinking.load() || aiMoveReady.load()) return;
        aiThinking.store(true);
        CheckersGame copy = game;

        std::thread t([&aiThinking, &aiMoveReady, &aiMoveResult, &aiMutex,
                       copy, aiIsWhiteLocal]() mutable {
            Move best = searchBestMoveStrong(copy, aiIsWhiteLocal, AI_TIME_LIMIT_MS);
            {
                std::lock_guard<std::mutex> lock(aiMutex);
                aiMoveResult = best;
            }
            aiMoveReady.store(true);
            aiThinking.store(false);
        });
        t.detach();
    };

    Move hintMove;
    bool needNewHint = true;

    sf::Clock frameClock;

    while (window.isOpen()) {
        float dt = frameClock.restart().asSeconds();

        sf::Event event;
        while (window.pollEvent(event)) {
            if (event.type == sf::Event::Closed) {
                window.close();
            }

            if (event.type == sf::Event::KeyPressed) {
                if (event.key.code == sf::Keyboard::Escape) {
                    window.close();
                }

                if (state == GameState::MENU) {
                    if (event.key.code == sf::Keyboard::Num1) {
                        twoPlayerMode = !twoPlayerMode;
                    }
                    if (event.key.code == sf::Keyboard::Num2) {
                        humanIsWhite = !humanIsWhite;
                    }
                    if (event.key.code == sf::Keyboard::H) {
                        hintsEnabled = !hintsEnabled;
                    }
                    if (event.key.code == sf::Keyboard::Z) {
                        playerColorIdx = (playerColorIdx - 1 + COLOR_COUNT) % COLOR_COUNT;
                    }
                    if (event.key.code == sf::Keyboard::X) {
                        playerColorIdx = (playerColorIdx + 1) % COLOR_COUNT;
                    }
                    if (event.key.code == sf::Keyboard::C) {
                        opponentColorIdx = (opponentColorIdx - 1 + COLOR_COUNT) % COLOR_COUNT;
                    }
                    if (event.key.code == sf::Keyboard::V) {
                        opponentColorIdx = (opponentColorIdx + 1) % COLOR_COUNT;
                    }

                    if (event.key.code == sf::Keyboard::Enter) {
                        aiIsWhite = !humanIsWhite;
                        resetGame();
                        aiThinking.store(false);
                        aiMoveReady.store(false);
                        hintMove.path.clear();
                        hintMove.captured.clear();
                        needNewHint = true;
                        state = GameState::PLAY;
                    }
                } else if (state == GameState::PLAY) {
                    if (event.key.code == sf::Keyboard::R) {
                        resetGame();
                        aiThinking.store(false);
                        aiMoveReady.store(false);
                        hintMove.path.clear();
                        hintMove.captured.clear();
                        needNewHint = true;
                    }
                    if (event.key.code == sf::Keyboard::C) {
                        if (!isAnimating) {
                            if (!snapshots.empty()) {
                                restoreFromSnapshot();
                                if (!snapshots.empty()) {
                                    restoreFromSnapshot();
                                }
                                hintMove.path.clear();
                                hintMove.captured.clear();
                                needNewHint = true;
                            }
                        }
                    }
                    if (event.key.code == sf::Keyboard::H) {
                        hintsEnabled = !hintsEnabled;
                    }
                    if (event.key.code == sf::Keyboard::T) {
                        twoPlayerMode = !twoPlayerMode;
                        aiIsWhite     = !humanIsWhite;
                        aiThinking.store(false);
                        aiMoveReady.store(false);
                        hintMove.path.clear();
                        hintMove.captured.clear();
                        needNewHint = true;
                    }
                }
            }

            if (state == GameState::PLAY &&
                event.type == sf::Event::MouseButtonPressed &&
                event.mouseButton.button == sf::Mouse::Left &&
                !isAnimating && !game.gameOver) {

                int mx = event.mouseButton.x;
                int my = event.mouseButton.y;

                const int boardTopY    = HEADER_H;
                const int boardBottomY = HEADER_H + BOARD_PX;
                const int boardLeftX   = 200;
                const int boardRightX  = 200 + BOARD_PX;

                if (mx >= boardLeftX && mx < boardRightX &&
                    my >= boardTopY   && my < boardBottomY) {

                    int x = (mx - boardLeftX) / TILE_SIZE;
                    int y = (my - boardTopY)  / TILE_SIZE;

                    if (selectedX == -1) {
                        int p = game.board[y][x];
                        if (p != 0) {
                            bool canSelect = false;
                            if (twoPlayerMode) {
                                if (game.whiteTurn && game.isWhite(p)) canSelect = true;
                                if (!game.whiteTurn && game.isBlack(p)) canSelect = true;
                            } else {
                                bool humanTurnNow = (game.whiteTurn == humanIsWhite);
                                if (humanTurnNow) {
                                    if (humanIsWhite && game.isWhite(p)) canSelect = true;
                                    if (!humanIsWhite && game.isBlack(p)) canSelect = true;
                                }
                            }
                            if (canSelect) {
                                selectedX = x;
                                selectedY = y;
                                movesFromSelected.clear();
                                vector<Move> all = game.getAllMoves(game.whiteTurn);
                                for (auto &m : all) {
                                    int sx = m.path.front().first;
                                    int sy = m.path.front().second;
                                    if (sx == x && sy == y)
                                        movesFromSelected.push_back(m);
                                }
                            }
                        }
                    } else {
                        bool moved = false;
                        for (auto &m : movesFromSelected) {
                            int tx = m.path.back().first;
                            int ty = m.path.back().second;
                            if (tx == x && ty == y) {
                                float elapsed = turnClock.getElapsedTime().asSeconds();
                                if (game.whiteTurn) whiteTotalTime += elapsed;
                                else               blackTotalTime += elapsed;
                                turnClock.restart();

                                pushSnapshot();

                                bool wasWhite = game.whiteTurn;
                                game.applyMove(m);
                                game.whiteTurn = !game.whiteTurn;
                                lastMove = m;

                                moveHistory.push_back(moveToStr(m, wasWhite));
                                if (moveHistory.size() > 60)
                                    moveHistory.erase(moveHistory.begin());

                                isAnimating = true;
                                animMove = m;
                                animSegmentIndex = 0;
                                animTime = 0.f;
                                int fx = m.path.back().first;
                                int fy = m.path.back().second;
                                animPieceValue = game.board[fy][fx];

                                moved = true;
                                break;
                            }
                        }
                        selectedX = selectedY = -1;
                        movesFromSelected.clear();

                        if (moved) {
                            turnClock.restart();
                            hintMove.path.clear();
                            hintMove.captured.clear();
                            needNewHint = true;
                        }
                    }
                }
            }
        }

        if (state == GameState::PLAY) {
            game.checkGameOver();

            if (isAnimating && animMove.path.size() >= 2) {
                animTime += dt;
                while (animTime >= SEG_DURATION && isAnimating) {
                    animTime -= SEG_DURATION;
                    animSegmentIndex++;
                    if (animSegmentIndex >= animMove.path.size() - 1) {
                        isAnimating = false;
                        animSegmentIndex = 0;
                        animTime = 0.f;
                    }
                }
            }

            if (!twoPlayerMode && !game.gameOver) {
                bool aiTurnNow = (game.whiteTurn == aiIsWhite);

                if (aiTurnNow && !isAnimating) {
                    if (!aiThinking.load() && !aiMoveReady.load()) {
                        startAIThinking(aiIsWhite);
                    }
                }

                if (aiMoveReady.load() && !isAnimating && !game.gameOver) {
                    bool stillAITurn = (game.whiteTurn == aiIsWhite);
                    if (stillAITurn) {
                        Move m;
                        {
                            std::lock_guard<std::mutex> lock(aiMutex);
                            m = aiMoveResult;
                        }
                        aiMoveReady.store(false);

                        if (m.path.size() >= 2) {
                            float elapsed = turnClock.getElapsedTime().asSeconds();
                            if (game.whiteTurn) whiteTotalTime += elapsed;
                            else               blackTotalTime += elapsed;
                            turnClock.restart();

                            pushSnapshot();

                            bool wasWhite = game.whiteTurn;
                            game.applyMove(m);
                            game.whiteTurn = !game.whiteTurn;
                            lastMove = m;

                            moveHistory.push_back(moveToStr(m, wasWhite));
                            if (moveHistory.size() > 60)
                                moveHistory.erase(moveHistory.begin());

                            isAnimating = true;
                            animMove = m;
                            animSegmentIndex = 0;
                            animTime = 0.f;
                            int fx = m.path.back().first;
                            int fy = m.path.back().second;
                            animPieceValue = game.board[fy][fx];

                            turnClock.restart();

                            hintMove.path.clear();
                            hintMove.captured.clear();
                            needNewHint = true;
                        } else {
                            game.checkGameOver();
                        }
                    }
                }
            }

            // Podpowiedź – liczona raz na turę, na tym samym poziomie co AI
            if (hintsEnabled &&
                !game.gameOver &&
                !isAnimating &&
                needNewHint) {

                hintMove = searchBestMoveStrong(game, game.whiteTurn, HINT_TIME_LIMIT_MS);
                needNewHint = false;
            }
        }

        int evalScore = game.evaluate();
        double evalForDisplay = evalScore / 100.0;

        sf::Color playerColor   = COLOR_OPTIONS[playerColorIdx].color;
        sf::Color opponentColor = COLOR_OPTIONS[opponentColorIdx].color;
        sf::Color colorWhitePieces;
        sf::Color colorBlackPieces;

        if (twoPlayerMode) {
            colorWhitePieces = playerColor;
            colorBlackPieces = opponentColor;
        } else {
            if (humanIsWhite) {
                colorWhitePieces = playerColor;
                colorBlackPieces = opponentColor;
            } else {
                colorWhitePieces = opponentColor;
                colorBlackPieces = playerColor;
            }
        }

        window.clear(sf::Color(30,30,30));

        float boardOffsetX = 200.f;
        float boardOffsetY = (float)HEADER_H;

        if (state == GameState::MENU) {
            if (font.getInfo().family != "") {
                sf::Text title;
                title.setFont(font);
                title.setCharacterSize(32);
                title.setFillColor(sf::Color::Cyan);
                title.setString("WARCABY – MENU STARTOWE");
                title.setPosition(80.f, 40.f);
                window.draw(title);

                sf::Text body;
                body.setFont(font);
                body.setCharacterSize(20);
                body.setFillColor(sf::Color::White);

                string txt;
                txt += "Ustawienia przed gra:\n\n";
                txt += "Tryb gry (1): ";
                txt += (twoPlayerMode ? "2 GRACZY\n" : "CZLOWIEK vs AI\n");

                txt += "Strona gracza (2): ";
                txt += (humanIsWhite ? "BIALE\n\n" : "CZARNE\n\n");

                txt += "Kolor pionkow gracza (Z/X): ";
                txt += COLOR_OPTIONS[playerColorIdx].name;
                txt += "\n";

                txt += "Kolor pionkow przeciwnika (C/V): ";
                txt += COLOR_OPTIONS[opponentColorIdx].name;
                txt += "\n\n";

                txt += "Podpowiedzi (H): ";
                txt += (hintsEnabled ? "WLACZONE\n\n" : "WYLACZONE\n\n");

                txt += "AI: gra bardzo silnie (iteracyjne poglebianie, wielowatkowo)\n\n";
                txt += "ENTER - start gry\n";
                txt += "ESC   - wyjscie\n";

                body.setString(txt);
                body.setPosition(80.f, 120.f);
                window.draw(body);

                float baseX  = body.getPosition().x;
                float baseY  = body.getPosition().y;
                float lineH  = 24.f;

                sf::CircleShape previewPlayer(12.f);
                previewPlayer.setFillColor(COLOR_OPTIONS[playerColorIdx].color);
                previewPlayer.setOutlineColor(sf::Color::Black);
                previewPlayer.setOutlineThickness(1.f);
                previewPlayer.setPosition(baseX + 360.f,
                                          baseY + 4.f*lineH - 6.f);
                window.draw(previewPlayer);

                sf::CircleShape previewOpp(12.f);
                previewOpp.setFillColor(COLOR_OPTIONS[opponentColorIdx].color);
                previewOpp.setOutlineColor(sf::Color::Black);
                previewOpp.setOutlineThickness(1.f);
                previewOpp.setPosition(baseX + 360.f,
                                       baseY + 6.f*lineH - 6.f);
                window.draw(previewOpp);
            }

        } else if (state == GameState::PLAY) {
            // plansza
            for (int y = 0; y < CheckersGame::SIZE; ++y) {
                for (int x = 0; x < CheckersGame::SIZE; ++x) {
                    sf::RectangleShape tile(sf::Vector2f(TILE_SIZE, TILE_SIZE));
                    tile.setPosition(boardOffsetX + x*TILE_SIZE,
                                     boardOffsetY + y*TILE_SIZE);
                    bool dark = ((x + y) % 2 == 1);
                    if (dark) tile.setFillColor(sf::Color(118,150,86));
                    else      tile.setFillColor(sf::Color(238,238,210));
                    window.draw(tile);
                }
            }

            if (font.getInfo().family != "") {
                for (int x = 0; x < CheckersGame::SIZE; ++x) {
                    char col = 'A' + x;
                    sf::Text tTop, tBottom;
                    tTop.setFont(font);
                    tTop.setCharacterSize(14);
                    tTop.setFillColor(sf::Color(0,200,0));
                    tTop.setString(string(1, col));
                    tTop.setPosition(boardOffsetX + x*TILE_SIZE + TILE_SIZE/2.f - 5.f,
                                     boardOffsetY - 24.f);
                    window.draw(tTop);

                    tBottom = tTop;
                    tBottom.setPosition(boardOffsetX + x*TILE_SIZE + TILE_SIZE/2.f - 5.f,
                                        boardOffsetY + BOARD_PX + 4.f);
                    window.draw(tBottom);
                }
                for (int y = 0; y < CheckersGame::SIZE; ++y) {
                    char row = '1' + (7 - y);
                    sf::Text tLeft, tRight;
                    tLeft.setFont(font);
                    tLeft.setCharacterSize(14);
                    tLeft.setFillColor(sf::Color(0,200,0));
                    tLeft.setString(string(1, row));
                    tLeft.setPosition(boardOffsetX - 24.f,
                                      boardOffsetY + y*TILE_SIZE + TILE_SIZE/2.f - 8.f);
                    window.draw(tLeft);

                    tRight = tLeft;
                    tRight.setPosition(boardOffsetX + BOARD_PX + 4.f,
                                       boardOffsetY + y*TILE_SIZE + TILE_SIZE/2.f - 8.f);
                    window.draw(tRight);
                }
            }

            int animFinalX = -1, animFinalY = -1;
            if (isAnimating && animMove.path.size() >= 1) {
                animFinalX = animMove.path.back().first;
                animFinalY = animMove.path.back().second;
            }

            // piony
            for (int y = 0; y < CheckersGame::SIZE; ++y) {
                for (int x = 0; x < CheckersGame::SIZE; ++x) {
                    if (isAnimating && x == animFinalX && y == animFinalY) continue;
                    int p = game.board[y][x];
                    if (p == 0) continue;

                    sf::CircleShape piece(TILE_SIZE/2 - 8);
                    piece.setPosition(boardOffsetX + x*TILE_SIZE + 8,
                                      boardOffsetY + y*TILE_SIZE + 8);
                    if (game.isWhite(p)) piece.setFillColor(colorWhitePieces);
                    else                 piece.setFillColor(colorBlackPieces);
                    piece.setOutlineThickness(2);
                    piece.setOutlineColor(sf::Color::Black);
                    window.draw(piece);

                    if (game.isKing(p)) {
                        sf::CircleShape crown(TILE_SIZE/2 - 20);
                        crown.setPosition(boardOffsetX + x*TILE_SIZE + 20,
                                          boardOffsetY + y*TILE_SIZE + 20);
                        crown.setFillColor(sf::Color(255,215,0,200));
                        window.draw(crown);
                    }
                }
            }

            if (isAnimating && animMove.path.size() >= 2) {
                int sx = animMove.path[animSegmentIndex].first;
                int sy = animMove.path[animSegmentIndex].second;
                int ex = animMove.path[animSegmentIndex+1].first;
                int ey = animMove.path[animSegmentIndex+1].second;

                float tseg = animTime / SEG_DURATION;
                if (tseg < 0.f) tseg = 0.f;
                if (tseg > 1.f) tseg = 1.f;

                float fromX = boardOffsetX + sx*TILE_SIZE + TILE_SIZE/2.f;
                float fromY = boardOffsetY + sy*TILE_SIZE + TILE_SIZE/2.f;
                float toX   = boardOffsetX + ex*TILE_SIZE + TILE_SIZE/2.f;
                float toY   = boardOffsetY + ey*TILE_SIZE + TILE_SIZE/2.f;

                float curX = fromX + (toX - fromX) * tseg;
                float curY = fromY + (toY - fromY) * tseg;

                sf::CircleShape piece(TILE_SIZE/2 - 8);
                piece.setPosition(curX - (TILE_SIZE/2 - 8),
                                  curY - (TILE_SIZE/2 - 8));
                if (game.isWhite(animPieceValue))
                    piece.setFillColor(colorWhitePieces);
                else
                    piece.setFillColor(colorBlackPieces);
                piece.setOutlineThickness(2);
                piece.setOutlineColor(sf::Color::Black);
                window.draw(piece);

                if (game.isKing(animPieceValue)) {
                    sf::CircleShape crown(TILE_SIZE/2 - 20);
                    crown.setPosition(curX - (TILE_SIZE/2 - 20),
                                      curY - (TILE_SIZE/2 - 20));
                    crown.setFillColor(sf::Color(255,215,0,200));
                    window.draw(crown);
                }
            }

            if (lastMove.path.size() >= 2) {
                sf::VertexArray trail(sf::LineStrip, lastMove.path.size());
                for (size_t i = 0; i < lastMove.path.size(); ++i) {
                    int x = lastMove.path[i].first;
                    int y = lastMove.path[i].second;
                    trail[i].position = sf::Vector2f(
                        boardOffsetX + x*TILE_SIZE + TILE_SIZE/2.f,
                        boardOffsetY + y*TILE_SIZE + TILE_SIZE/2.f
                    );
                    trail[i].color = sf::Color(255,255,0,200);
                }
                window.draw(trail);

                for (auto &pt : lastMove.path) {
                    int x = pt.first;
                    int y = pt.second;
                    sf::CircleShape dot(6.f);
                    dot.setFillColor(sf::Color(255,255,0,220));
                    dot.setPosition(boardOffsetX + x*TILE_SIZE + TILE_SIZE/2.f - 6.f,
                                    boardOffsetY + y*TILE_SIZE + TILE_SIZE/2.f - 6.f);
                    window.draw(dot);
                }
            }

            if (selectedX != -1) {
                sf::RectangleShape sel(sf::Vector2f(TILE_SIZE, TILE_SIZE));
                sel.setPosition(boardOffsetX + selectedX*TILE_SIZE,
                                boardOffsetY + selectedY*TILE_SIZE);
                sel.setFillColor(sf::Color(255,255,0,100));
                window.draw(sel);

                for (auto &m : movesFromSelected) {
                    int tx = m.path.back().first;
                    int ty = m.path.back().second;
                    sf::RectangleShape hl(sf::Vector2f(TILE_SIZE, TILE_SIZE));
                    hl.setPosition(boardOffsetX + tx*TILE_SIZE,
                                   boardOffsetY + ty*TILE_SIZE);
                    hl.setFillColor(sf::Color(0,0,255,80));
                    window.draw(hl);
                }
            }

            if (hintsEnabled &&
                !game.gameOver &&
                !isAnimating &&
                hintMove.path.size() >= 2) {

                int fx = hintMove.path.front().first;
                int fy = hintMove.path.front().second;
                int tx = hintMove.path.back().first;
                int ty = hintMove.path.back().second;

                sf::Vector2f from(
                    boardOffsetX + fx*TILE_SIZE + TILE_SIZE/2.f,
                    boardOffsetY + fy*TILE_SIZE + TILE_SIZE/2.f
                );
                sf::Vector2f to(
                    boardOffsetX + tx*TILE_SIZE + TILE_SIZE/2.f,
                    boardOffsetY + ty*TILE_SIZE + TILE_SIZE/2.f
                );

                sf::VertexArray line(sf::Lines, 2);
                line[0].position = from;
                line[0].color = sf::Color::Red;
                line[1].position = to;
                line[1].color = sf::Color::Red;
                window.draw(line);

                sf::Vector2f dir = to - from;
                float len = std::sqrt(dir.x*dir.x + dir.y*dir.y);
                if (len > 1.f) {
                    sf::Vector2f udir = dir / len;
                    sf::Vector2f base = to - udir*20.f;
                    sf::Vector2f perp(-udir.y, udir.x);
                    float w = 7.f;

                    sf::ConvexShape arrowHead;
                    arrowHead.setPointCount(3);
                    arrowHead.setPoint(0, to);
                    arrowHead.setPoint(1, base + perp*w);
                    arrowHead.setPoint(2, base - perp*w);
                    arrowHead.setFillColor(sf::Color::Red);
                    window.draw(arrowHead);
                }
            }

            // panel lewej historii
            if (font.getInfo().family != "") {
                sf::Text histTitle;
                histTitle.setFont(font);
                histTitle.setCharacterSize(18);
                histTitle.setFillColor(sf::Color::White);
                histTitle.setString("Historia ruchow:");
                histTitle.setPosition(10.f, HEADER_H + 10.f);
                window.draw(histTitle);

                sf::Text hist;
                hist.setFont(font);
                hist.setCharacterSize(14);
                hist.setFillColor(sf::Color::White);

                string htxt;
                int maxLines = 20;
                int start = (moveHistory.size() > (size_t)maxLines)
                            ? (int)moveHistory.size() - maxLines
                            : 0;
                for (size_t i = start; i < moveHistory.size(); ++i) {
                    htxt += moveHistory[i] + "\n";
                }
                hist.setString(htxt);
                hist.setPosition(10.f, HEADER_H + 40.f);
                window.draw(hist);
            }

            // panel prawy
            if (font.getInfo().family != "") {
                sf::Text info;
                info.setFont(font);
                info.setCharacterSize(15);
                info.setFillColor(sf::Color::White);

                float currentTurnTime = turnClock.getElapsedTime().asSeconds();
                float wTimeShow = whiteTotalTime + (game.whiteTurn && !game.gameOver ? currentTurnTime : 0.f);
                float bTimeShow = blackTotalTime + (!game.whiteTurn && !game.gameOver ? currentTurnTime : 0.f);

                string txt;
                txt += "Tryb: ";
                txt += (twoPlayerMode ? "2 GRACZY\n" : "CZLOWIEK vs AI\n");
                txt += "Ruch: ";
                txt += (game.whiteTurn ? "BIALE\n\n" : "CZARNE\n\n");

                txt += "Czas Bialych: " + formatTime(wTimeShow) + "\n";
                txt += "Czas Czarnych: " + formatTime(bTimeShow) + "\n\n";

                txt += "AI: iteracyjne poglebianie, wielowatkowo, limit ~10s\n";
                txt += "Podpowiedzi (H): ";
                txt += (hintsEnabled ? "WLACZONE\n\n" : "WYLACZONE\n\n");

                txt += "Kolor pionkow gracza: ";
                txt += COLOR_OPTIONS[playerColorIdx].name;
                txt += "\nKolor pionkow przeciwnika: ";
                txt += COLOR_OPTIONS[opponentColorIdx].name;
                txt += "\n\n";

                txt += "Sterowanie:\n";
                txt += "LPM - ruch\n";
                txt += "R   - restart\n";
                txt += "C   - cofnij swoj ostatni ruch\n";
                txt += "H   - wlacz/wylacz podpowiedzi\n";
                txt += "T   - vs AI / 2 graczy\n";
                txt += "ESC - wyjscie\n\n";

                if (game.gameOver) {
                    if (game.winner == 1)      txt += "KONIEC: WYGRALY BIALE\n";
                    else if (game.winner == -1) txt += "KONIEC: WYGRALY CZARNE\n";
                    else                        txt += "KONIEC: REMIS\n";
                } else if (!twoPlayerMode && aiThinking.load()) {
                    txt += "AI: licze ruch...\n";
                }

                info.setString(txt);
                info.setPosition(boardOffsetX + BOARD_PX + 70.f, HEADER_H + 10.f);
                window.draw(info);
            }

            // pasek oceny
            if (font.getInfo().family != "") {
                float barW = 20.f;
                float barH = 60.f;
                float barX = 200.f + BOARD_PX/2.f - barW/2.f;
                float barY = (HEADER_H - barH)/2.f;

                sf::RectangleShape barBg(sf::Vector2f(barW, barH));
                barBg.setPosition(barX, barY);
                barBg.setFillColor(sf::Color(50,50,50));
                barBg.setOutlineColor(sf::Color::White);
                barBg.setOutlineThickness(1.f);
                window.draw(barBg);

                double norm = evalScore / 1000.0;
                if (norm < -1.0) norm = -1.0;
                if (norm >  1.0) norm =  1.0;

                float half    = barH / 2.f;
                float centerY = barY + half;

                if (norm > 0) {
                    float h = half * (float)norm;
                    sf::RectangleShape bar(sf::Vector2f(barW, h));
                    bar.setPosition(barX, centerY - h);
                    bar.setFillColor(sf::Color(0,200,0));
                    window.draw(bar);
                } else if (norm < 0) {
                    float h = half * (float)(-norm);
                    sf::RectangleShape bar(sf::Vector2f(barW, h));
                    bar.setPosition(barX, centerY);
                    bar.setFillColor(sf::Color(200,0,0));
                    window.draw(bar);
                }

                sf::RectangleShape mid(sf::Vector2f(barW, 2.f));
                mid.setPosition(barX, centerY - 1.f);
                mid.setFillColor(sf::Color::White);
                window.draw(mid);

                sf::Text evalText;
                evalText.setFont(font);
                evalText.setCharacterSize(16);
                evalText.setFillColor(sf::Color::White);
                char buf[64];
                snprintf(buf, sizeof(buf), "%+.2f", evalForDisplay);
                string msg = string("Ocena: ") + buf + " ( >0 dla bialych )";
                evalText.setString(msg);
                evalText.setPosition(barX + barW + 10.f, barY + barH/2.f - 10.f);
                window.draw(evalText);
            }

            if (font.getInfo().family != "" && game.gameOver) {
                sf::Text center;
                center.setFont(font);
                center.setCharacterSize(32);
                center.setFillColor(sf::Color::Red);
                string cmsg;
                if (game.winner == 1)      cmsg = "WYGRALY BIALE";
                else if (game.winner == -1) cmsg = "WYGRALY CZARNE";
                else                        cmsg = "REMIS";
                center.setString(cmsg);
                center.setPosition(boardOffsetX + BOARD_PX/2.f - 160.f,
                                   boardOffsetY + BOARD_PX/2.f - 20.f);
                window.draw(center);
            }
        }

        window.display();
    }

    return 0;
}
