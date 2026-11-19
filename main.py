#!/usr/bin/env python3
"""
Mini NPU Simulator
==================
AI 가속기(NPU)의 핵심 연산인 MAC(Multiply-Accumulate)을 순수 Python으로 흉내 내는
콘솔 프로그램입니다.

  모드 1) 사용자 입력 (3x3)  : 필터 A/B와 패턴을 직접 입력해 판정
  모드 2) data.json 분석     : 5x5 / 13x13 / 25x25 케이스 일괄 판정 + PASS/FAIL 리포트
  모드 3) 패턴 생성기 (보너스): N 입력 시 Cross/X 패턴 자동 생성 후 판정·성능 측정

실행 환경: Python 3.8+, 표준 라이브러리만 사용 (NumPy 등 외부 라이브러리 미사용)
"""

import argparse
import json
import math
import os
import re
import sys
import time
import unicodedata
from typing import Callable, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# 상수 / 정책
# ---------------------------------------------------------------------------
EPSILON = 1e-9            # 점수 비교 허용오차: |a - b| < EPSILON 이면 동점
REPEAT = 10               # 성능 측정 반복 횟수: 요구사항대로 MAC 1회 호출을 10번 측정
BENCH_SIZES = (3, 5, 13, 25)
PREVIEW_LIMIT = 25        # 이 크기 이하만 ■□ 미리보기 출력
GENERATOR_MAX = 101       # 패턴 생성기 최대 크기 (측정 시간이 과도해지지 않도록)

LABEL_CROSS = "Cross"
LABEL_X = "X"
UNDECIDED = "UNDECIDED"

# 외부 표기 → 표준 라벨. 비교 전에 strip().lower() 처리하므로 대소문자 무관.
LABEL_ALIASES = {
    "+": LABEL_CROSS,
    "cross": LABEL_CROSS,
    "x": LABEL_X,
}

FILTER_KEY_RE = re.compile(r"^size_(\d+)$")
PATTERN_KEY_RE = re.compile(r"^size_(\d+)_(\d+)$")

# 실패 원인 분류 (데이터/스키마 vs 로직 vs 수치 비교)
CAT_SCHEMA = "데이터/스키마"
CAT_NUMERIC = "수치 비교(동점)"
CAT_MISMATCH = "판정 불일치"

LINE_WIDTH = 60


# ---------------------------------------------------------------------------
# 예외
# ---------------------------------------------------------------------------
class SchemaError(Exception):
    """data.json 구조/값/크기 문제. 케이스 단위 FAIL 처리에 사용."""


class InputFormatError(Exception):
    """콘솔 입력 형식 문제. 재입력 유도에 사용."""


class UserExit(Exception):
    """입력 스트림 종료(EOF) 등으로 프로그램을 정상 종료해야 할 때."""


# ---------------------------------------------------------------------------
# 데이터 구조: N x N 격자
# ---------------------------------------------------------------------------
def to_number(value) -> float:
    """JSON 값 하나를 float로 변환. bool / 문자열 / NaN / inf 는 거부."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("숫자 타입이 아닙니다")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("유한한 숫자가 아닙니다")
    return number


class Grid:
    """N x N 크기의 2차원 숫자 배열 (패턴과 필터 공용 자료구조)."""

    __slots__ = ("n", "data")

    def __init__(self, n: int, fill: float = 0.0):
        if not isinstance(n, int) or n < 1:
            raise ValueError("격자 크기는 1 이상의 정수여야 합니다.")
        self.n = n
        self.data = [[float(fill)] * n for _ in range(n)]

    # --- 특정 위치 값 저장/읽기 -------------------------------------------
    def get(self, row: int, col: int) -> float:
        self._check_index(row, col)
        return self.data[row][col]

    def set(self, row: int, col: int, value) -> None:
        self._check_index(row, col)
        self.data[row][col] = to_number(value)

    def _check_index(self, row: int, col: int) -> None:
        if not (0 <= row < self.n and 0 <= col < self.n):
            raise IndexError(
                "위치 ({}, {})는 {}x{} 범위를 벗어났습니다.".format(row, col, self.n, self.n)
            )

    # --- 생성/변환 ---------------------------------------------------------
    @classmethod
    def from_rows(cls, rows, name: str = "배열") -> "Grid":
        """2차원 리스트를 검증하며 Grid로 변환. 문제가 있으면 SchemaError."""
        if not isinstance(rows, list) or len(rows) == 0:
            raise SchemaError("{}: 비어 있지 않은 2차원 리스트가 아닙니다.".format(name))
        n = len(rows)
        grid = cls(n)
        for r, row in enumerate(rows):
            if not isinstance(row, list):
                raise SchemaError("{}: {}번째 행이 리스트가 아닙니다.".format(name, r + 1))
            if len(row) != n:
                raise SchemaError(
                    "{}: 정사각형이 아닙니다 ({}행의 열 수 {} ≠ 행 수 {}).".format(
                        name, r + 1, len(row), n)
                )
            for c, value in enumerate(row):
                try:
                    grid.data[r][c] = to_number(value)
                except ValueError as err:
                    raise SchemaError(
                        "{}: ({},{}) 위치 값 {!r} - {}.".format(name, r + 1, c + 1, value, err)
                    )
        return grid

    def flatten(self) -> List[float]:
        """행 우선(row-major) 순서의 1차원 리스트(길이 N²)로 변환."""
        return [value for row in self.data for value in row]

    @property
    def cells(self) -> int:
        return self.n * self.n


# ---------------------------------------------------------------------------
# MAC 연산 (반복문 직접 구현)
# ---------------------------------------------------------------------------
def mac_2d(pattern: Grid, filt: Grid) -> float:
    """기본 구현: 2차원 인덱스 [i][j]로 같은 위치끼리 곱해 누적."""
    if pattern.n != filt.n:
        raise SchemaError("크기 불일치: 패턴 {0}x{0}, 필터 {1}x{1}".format(pattern.n, filt.n))
    a = pattern.data
    b = filt.data
    n = pattern.n
    total = 0.0
    for i in range(n):
        for j in range(n):
            total += a[i][j] * b[i][j]
    return total


def mac_flat(pattern_flat: List[float], filter_flat: List[float]) -> float:
    """보너스 최적화: 1차원 배열(길이 N²)을 인덱스 하나로 순회.
    합산 순서가 mac_2d와 동일(row-major)하므로 결과 값도 비트 단위로 같다."""
    if len(pattern_flat) != len(filter_flat):
        raise SchemaError("크기 불일치: 길이 {} vs {}".format(len(pattern_flat), len(filter_flat)))
    total = 0.0
    for k in range(len(pattern_flat)):
        total += pattern_flat[k] * filter_flat[k]
    return total


# ---------------------------------------------------------------------------
# 점수 비교 정책 / 라벨 정규화
# ---------------------------------------------------------------------------
def compare_scores(score_a: float, score_b: float, eps: float = EPSILON) -> int:
    """a가 크면 1, b가 크면 -1, |a-b| < eps 이면 0(동점)."""
    if abs(score_a - score_b) < eps:
        return 0
    return 1 if score_a > score_b else -1


def naive_compare(score_a: float, score_b: float) -> int:
    """허용오차 없이 비교했을 때의 결과 (리포트에서 대조용으로만 사용)."""
    if score_a == score_b:
        return 0
    return 1 if score_a > score_b else -1


def normalize_label(raw) -> Optional[str]:
    """'+', 'cross', 'x' 등 외부 표기를 표준 라벨(Cross/X)로. 알 수 없으면 None."""
    if not isinstance(raw, str):
        return None
    return LABEL_ALIASES.get(raw.strip().lower())


def verdict_from(cross_score: float, x_score: float, eps: float = EPSILON) -> str:
    result = compare_scores(cross_score, x_score, eps)
    if result == 0:
        return UNDECIDED
    return LABEL_CROSS if result > 0 else LABEL_X


# ---------------------------------------------------------------------------
# 패턴 생성기 (보너스)
# ---------------------------------------------------------------------------
def generate_pattern(label: str, n: int) -> Grid:
    """표준 라벨(Cross/X)과 크기 N으로 0/1 패턴을 생성. N은 3 이상 홀수."""
    if n < 3 or n % 2 == 0:
        raise ValueError("패턴 크기는 3 이상의 홀수여야 합니다 (중심선이 1칸이어야 함).")
    grid = Grid(n)
    center = n // 2
    for i in range(n):
        if label == LABEL_CROSS:
            grid.set(center, i, 1)
            grid.set(i, center, 1)
        elif label == LABEL_X:
            grid.set(i, i, 1)
            grid.set(i, n - 1 - i, 1)
        else:
            raise ValueError("알 수 없는 라벨: {}".format(label))
    return grid


# ---------------------------------------------------------------------------
# 성능 측정
# ---------------------------------------------------------------------------
def measure_ms(func: Callable, *args, repeat: int = REPEAT) -> float:
    """MAC 함수 1회 호출 시간을 repeat(=10)번 측정해 평균(ms)을 반환.

    - 측정 구간은 함수 호출 한 번뿐이다(입력/출력/파일 읽기 제외).
    - 첫 호출은 워밍업으로 실행만 하고 측정에는 넣지 않는다.
    """
    func(*args)  # 워밍업 (측정 제외)
    elapsed = 0.0
    for _ in range(repeat):
        start = time.perf_counter()
        func(*args)
        elapsed += time.perf_counter() - start
    return elapsed / repeat * 1000.0


def display_width(text: str) -> int:
    """터미널 표시 폭: 한글 등 전각 문자는 2칸으로 계산."""
    return sum(2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1 for ch in text)


def pad(text: str, width: int, align: str = ">") -> str:
    gap = max(0, width - display_width(text))
    return " " * gap + text if align == ">" else text + " " * gap


def table_row(cells: List[str], widths: List[int], aligns: str) -> str:
    return " | ".join(pad(c, w, a) for c, w, a in zip(cells, widths, aligns)).rstrip()


def run_benchmark(samples: List[Tuple[int, Grid, Grid, str]], repeat: int) -> None:
    """samples: (크기, 패턴, 필터, 데이터 출처) 목록으로 성능 표 2종을 출력."""
    rows = []
    for n, pattern, filt, source in samples:
        flat_p, flat_f = pattern.flatten(), filt.flatten()   # 변환 비용은 측정에서 제외
        t2d = measure_ms(mac_2d, pattern, filt, repeat=repeat)
        t1d = measure_ms(mac_flat, flat_p, flat_f, repeat=repeat)
        rows.append((n, t2d, t1d, source))

    print("측정 방식: MAC 함수 1회 호출을 perf_counter로 {}회 측정한 평균".format(repeat))
    print("          (워밍업 1회 제외, 입력/출력/파일 읽기 시간 미포함)\n")

    widths = [7, 13, 13, 10, 9, 9]
    aligns = ">>>>><"
    print(table_row(["크기", "연산 횟수(N²)", "평균 시간(ms)", "원소당(ns)", "3x3 대비", "데이터"],
                    widths, aligns))
    print("-" * (sum(widths) + 3 * (len(widths) - 1)))
    base_time = rows[0][1]
    for n, t2d, _, source in rows:
        ops = n * n
        ratio = t2d / base_time if base_time > 0 else float("nan")
        print(table_row([
            "{0}x{0}".format(n), "{:,}".format(ops), "{:.5f}".format(t2d),
            "{:.1f}".format(t2d * 1e6 / ops), "{:.1f}x".format(ratio), source,
        ], widths, aligns))

    if len(rows) > 1:
        (n0, t0, _, _), (n1, t1, _, _) = rows[0], rows[-1]
        print("\n해석: {0}x{0} → {1}x{1}: 연산 횟수 {2:.1f}배 / 시간 {3:.1f}배".format(
            n0, n1, (n1 * n1) / (n0 * n0), t1 / t0 if t0 > 0 else float("nan")))
        if len(rows) > 2:
            (na, ta, _, _), (nb, tb, _, _) = rows[-2], rows[-1]
            print("      {0}x{0} → {1}x{1}: 연산 횟수 {2:.1f}배 / 시간 {3:.1f}배".format(
                na, nb, (nb * nb) / (na * na), tb / ta if ta > 0 else float("nan")))
        print("      작은 N은 함수 호출 같은 고정 비용 비중이 커서 배율이 작게 보이고,")
        print("      N이 커질수록 시간 증가율이 N² 증가율에 가까워짐 → O(N²)")
        print("      (µs 단위 측정이라 실행할 때마다 값이 조금씩 달라질 수 있음)")

    print("\n[보너스] 메모리 접근 최적화: 2차원 a[i][j] vs 1차원 a[k] (동일 입력·동일 반복)")
    widths2 = [7, 11, 11, 8]
    print(table_row(["크기", "2D (ms)", "1D (ms)", "개선율"], widths2, ">>>>"))
    print("-" * (sum(widths2) + 3 * (len(widths2) - 1)))
    for n, t2d, t1d, _ in rows:
        gain = (t2d - t1d) / t2d * 100 if t2d > 0 else 0.0
        print(table_row(["{0}x{0}".format(n), "{:.5f}".format(t2d), "{:.5f}".format(t1d),
                         "{:+.1f}%".format(gain)], widths2, ">>>>"))


# ---------------------------------------------------------------------------
# 출력 도우미
# ---------------------------------------------------------------------------
def banner(step: str, title: str) -> None:
    print("\n" + "=" * LINE_WIDTH)
    print(" {} | {}".format(step, title))
    print("=" * LINE_WIDTH)


def fmt_num(value: float) -> str:
    """정수 값은 '1', 아니면 짧게 표시."""
    return str(int(value)) if value == int(value) else "{:g}".format(value)


def print_side_by_side(grids: List[Tuple[str, Grid]]) -> None:
    """여러 격자를 숫자 + ■□ 모양으로 나란히 출력 (저장 확인용)."""
    cell_w = max(len(fmt_num(v)) for _, g in grids for row in g.data for v in row)
    blocks = []
    for name, g in grids:
        lines = [name]
        for row in g.data:
            nums = " ".join(fmt_num(v).rjust(cell_w) for v in row)
            shape = "".join("■" if v > 0 else "□" for v in row)
            lines.append("{}   {}".format(nums, shape))
        blocks.append(lines)
    width = max(len(line) for block in blocks for line in block) + 4
    for i in range(max(len(b) for b in blocks)):
        print("  " + "".join((b[i] if i < len(b) else "").ljust(width) for b in blocks).rstrip())


def print_shape(grid: Grid, indent: str = "  ") -> None:
    for row in grid.data:
        print(indent + " ".join("■" if v > 0 else "□" for v in row))


# ---------------------------------------------------------------------------
# 콘솔 입력
# ---------------------------------------------------------------------------
def ask(prompt: str) -> str:
    try:
        return input(prompt)
    except EOFError:
        raise UserExit()


def parse_row(line: str, n: int) -> List[float]:
    tokens = line.split()
    if not tokens:
        raise InputFormatError("빈 줄입니다.")
    values = []
    for tok in tokens:
        try:
            value = float(tok)
        except ValueError:
            raise InputFormatError("'{}'은(는) 숫자로 읽을 수 없습니다.".format(tok))
        if not math.isfinite(value):
            raise InputFormatError("'{}'은(는) 유한한 숫자가 아닙니다.".format(tok))
        values.append(value)
    if len(values) != n:
        hint = ""
        if len(values) > n and len(values) % n == 0:
            hint = " (여러 행을 한 줄에 입력한 것 같습니다)"
        elif len(tokens) == 1 and tokens[0].isdigit() and len(tokens[0]) == n:
            spaced = " ".join(tokens[0])
            hint = " 숫자 사이에 띄어쓰기가 필요합니다. 예: '{}' → '{}'".format(tokens[0], spaced)
        raise InputFormatError("숫자 {}개가 입력되었습니다.{}".format(len(values), hint))
    return values


def read_grid(name: str, n: int = 3) -> Grid:
    """행을 한 줄씩 입력받고, 빈 줄(Enter)로 입력을 마친다.

    입력이 끝나면 행 수와 각 행의 열 수·숫자 형식을 한꺼번에 검사하고,
    하나라도 틀리면 오류를 모두 보여 준 뒤 그 행렬 전체를 다시 입력받는다.
    (정확히 n줄만 읽으면 n+1번째 줄이 다음 입력으로 넘어가 행 수 초과를 잡을 수 없기 때문)
    """
    while True:
        print("\n>> {} 입력: 각 줄에 숫자 {}개를 공백으로 구분, {}줄 입력 후 빈 줄에서 Enter"
              .format(name, n, n))
        print("   (첫 줄에 'cross'(또는 '+') / 'x' 를 입력하면 패턴 생성기로 자동 생성)")
        lines: List[str] = []
        while True:
            line = ask("  {}행 > ".format(len(lines) + 1))
            if not line.strip():
                break
            if not lines:
                label = normalize_label(line)
                if label:
                    grid = generate_pattern(label, n)
                    print("  -> {} 패턴 자동 생성:".format(label))
                    print_shape(grid, indent="     ")
                    return grid
            lines.append(line)

        errors: List[str] = []
        if len(lines) != n:
            errors.append("행 수 오류: {}줄이 필요한데 {}줄이 입력되었습니다.".format(n, len(lines)))
        rows: List[List[float]] = []
        for idx, line in enumerate(lines, start=1):
            try:
                rows.append(parse_row(line, n))
            except InputFormatError as err:
                errors.append("{}행: {}".format(idx, err))

        if not errors:
            return Grid.from_rows(rows, name)

        print("  [입력 형식 오류] 각 줄에 {}개의 숫자를 공백으로 구분해 {}줄 입력하세요.".format(n, n))
        for msg in errors:
            print("    - {}".format(msg))
        print("  -> {}를 처음부터 다시 입력해 주세요.".format(name))


def ask_choice(prompt: str, valid: Tuple[str, ...]) -> str:
    while True:
        choice = ask(prompt).strip()
        if choice in valid:
            return choice
        print("  [안내] {} 중 하나를 입력하세요.".format(", ".join(valid)))


# ---------------------------------------------------------------------------
# 모드 1: 사용자 입력 (3x3)
# ---------------------------------------------------------------------------
def run_user_mode(repeat: int) -> None:
    n = 3

    banner("STEP 1", "필터 입력 (3x3)")
    filter_a = read_grid("필터 A", n)
    filter_b = read_grid("필터 B", n)

    print("\n[저장 확인] 두 필터가 아래와 같이 저장되었습니다.")
    print_side_by_side([("필터 A", filter_a), ("필터 B", filter_b)])

    banner("STEP 2", "패턴 입력 (3x3)")
    pattern = read_grid("패턴", n)

    banner("STEP 3", "MAC 연산 및 판정")
    score_a = mac_2d(pattern, filter_a)
    score_b = mac_2d(pattern, filter_b)
    time_a = measure_ms(mac_2d, pattern, filter_a, repeat=repeat)
    time_b = measure_ms(mac_2d, pattern, filter_b, repeat=repeat)
    diff = abs(score_a - score_b)

    print("  필터 A 점수 : {!r}".format(score_a))
    print("  필터 B 점수 : {!r}".format(score_b))
    print("  점수 차이   : {:.3g}".format(diff))
    print("  연산 시간   : A {:.5f} ms / B {:.5f} ms  (MAC 1회, {}회 측정 평균)".format(
        time_a, time_b, repeat))

    result = compare_scores(score_a, score_b)
    if result == 0:
        print("  >> 판정    : 판정 불가 (|A-B| = {:.3g} < {:g}, 동점 처리)".format(diff, EPSILON))
    else:
        winner = "A" if result > 0 else "B"
        print("  >> 판정    : {}  (패턴이 필터 {}와 더 유사)".format(winner, winner))

    banner("STEP 4", "성능 분석 (3x3)")
    run_benchmark([(n, pattern, filter_a, "사용자 입력")], repeat)


# ---------------------------------------------------------------------------
# 모드 2: data.json 분석
# ---------------------------------------------------------------------------
def load_json(path: str):
    try:
        with open(path, "r", encoding="utf-8-sig") as fp:
            return json.load(fp), None
    except FileNotFoundError:
        return None, "파일을 찾을 수 없습니다: {}".format(path)
    except json.JSONDecodeError as err:
        return None, "JSON 형식 오류 ({}행 {}열): {}".format(err.lineno, err.colno, err.msg)
    except OSError as err:
        return None, "파일을 읽을 수 없습니다: {}".format(err)


def load_filters(raw) -> Tuple[Dict[int, Dict[str, Grid]], Dict[int, str]]:
    """filters 섹션을 읽어 {N: {'Cross': Grid, 'X': Grid}} 와 {N: 오류메시지} 반환."""
    loaded: Dict[int, Dict[str, Grid]] = {}
    errors: Dict[int, str] = {}

    if not isinstance(raw, dict):
        print("  [ERR] 'filters' 항목이 없거나 객체(dict) 형태가 아닙니다.")
        return loaded, errors

    for key, entry in raw.items():
        match = FILTER_KEY_RE.match(str(key))
        if not match:
            print("  [WARN] '{}': 'size_N' 형식이 아닌 필터 키라서 건너뜁니다.".format(key))
            continue
        n = int(match.group(1))
        try:
            if not isinstance(entry, dict):
                raise SchemaError("필터 묶음이 객체(dict) 형태가 아닙니다.")
            grids: Dict[str, Grid] = {}
            mapping = []
            for sub_key, rows in entry.items():
                label = normalize_label(sub_key)
                if label is None:
                    print("  [WARN] {}: 알 수 없는 필터 키 '{}' 무시".format(key, sub_key))
                    continue
                if label in grids:
                    raise SchemaError("'{}'가 정규화 후 {} 라벨과 중복됩니다.".format(sub_key, label))
                grid = Grid.from_rows(rows, "{}.{}".format(key, sub_key))
                if grid.n != n:
                    raise SchemaError("키는 {0}x{0}인데 '{1}' 필터는 {2}x{2}입니다.".format(
                        n, sub_key, grid.n))
                grids[label] = grid
                mapping.append("{}→{}".format(sub_key, label))
            missing = [lb for lb in (LABEL_CROSS, LABEL_X) if lb not in grids]
            if missing:
                raise SchemaError("{} 필터가 없습니다.".format("/".join(missing)))
            loaded[n] = grids
            print("  [OK]  {:<8} {}x{} 필터 2종 로드  (키 정규화: {})".format(
                key, n, n, ", ".join(mapping)))
        except SchemaError as err:
            errors[n] = str(err)
            print("  [ERR] {:<8} {}".format(key, err))
    return loaded, errors


def evaluate_case(case_id: str, entry, filters: Dict[int, Dict[str, Grid]],
                  filter_errors: Dict[int, str]) -> dict:
    """패턴 1건을 검증·연산·판정. 어떤 문제가 생겨도 예외 대신 결과 dict로 반환."""
    result = {
        "id": case_id, "n": None, "expected": None, "expected_raw": None,
        "cross": None, "x": None, "verdict": None, "naive": None,
        "status": "FAIL", "category": None, "reason": "",
    }
    try:
        match = PATTERN_KEY_RE.match(str(case_id))
        if not match:
            raise SchemaError("키가 'size_{N}_{idx}' 형식이 아닙니다.")
        n = int(match.group(1))
        result["n"] = n

        if not isinstance(entry, dict):
            raise SchemaError("패턴 항목이 객체(dict) 형태가 아닙니다.")
        if "input" not in entry:
            raise SchemaError("'input' 필드가 없습니다.")
        if "expected" not in entry:
            raise SchemaError("'expected' 필드가 없습니다.")

        result["expected_raw"] = entry["expected"]
        expected = normalize_label(entry["expected"])
        if expected is None:
            raise SchemaError("expected 값 {!r}을(를) 표준 라벨로 바꿀 수 없습니다.".format(
                entry["expected"]))
        result["expected"] = expected

        if n not in filters:
            reason = filter_errors.get(n, "data.json에 size_{} 필터가 없습니다.".format(n))
            raise SchemaError("size_{} 필터 사용 불가 - {}".format(n, reason))

        pattern = Grid.from_rows(entry["input"], "{}.input".format(case_id))
        if pattern.n != n:
            raise SchemaError("크기 불일치: 키는 {0}x{0}, input은 {1}x{1}".format(n, pattern.n))

        cross_score = mac_2d(pattern, filters[n][LABEL_CROSS])
        x_score = mac_2d(pattern, filters[n][LABEL_X])
        result["cross"], result["x"] = cross_score, x_score
        result["verdict"] = verdict_from(cross_score, x_score)
        naive = naive_compare(cross_score, x_score)
        result["naive"] = UNDECIDED if naive == 0 else (LABEL_CROSS if naive > 0 else LABEL_X)

        if result["verdict"] == expected:
            result["status"] = "PASS"
        elif result["verdict"] == UNDECIDED:
            result["category"] = CAT_NUMERIC
            result["reason"] = "점수 차 {:.2e} < {:g} → 동점(UNDECIDED) 규칙에 따라 FAIL".format(
                abs(cross_score - x_score), EPSILON)
        else:
            result["category"] = CAT_MISMATCH
            result["reason"] = "판정 {} ≠ expected {}".format(result["verdict"], expected)
    except SchemaError as err:
        result["category"] = CAT_SCHEMA
        result["reason"] = str(err)
    except Exception as err:  # 예상 못 한 오류도 케이스 단위로 격리
        result["category"] = CAT_SCHEMA
        result["reason"] = "처리 중 예외 발생: {}: {}".format(type(err).__name__, err)
    return result


def print_case(res: dict) -> None:
    size = "{0}x{0}".format(res["n"]) if res["n"] else "?"
    expected = res["expected"] or "?"
    raw = "'{}'→".format(res["expected_raw"]) if res["expected_raw"] is not None else ""
    print("\n[{}]  {} | expected {}{}".format(res["id"], size, raw, expected))
    if res["cross"] is not None:
        print("    Cross 점수 : {!r}".format(res["cross"]))
        print("    X 점수     : {!r}".format(res["x"]))
        print("    판정       : {}".format(res["verdict"]))
    mark = res["status"]
    tail = "" if res["status"] == "PASS" else "  ({}: {})".format(res["category"], res["reason"])
    print("    결과       : {}{}".format(mark, tail))


def collect_bench_samples(filters, patterns_raw) -> List[Tuple[int, Grid, Grid, str]]:
    """성능 측정용 입력 선택: data.json에 있으면 그 데이터를, 없으면 생성기 데이터를 사용."""
    samples = []
    for n in BENCH_SIZES:
        pattern = filt = None
        source = "생성기"
        if n in filters and isinstance(patterns_raw, dict):
            for key, entry in patterns_raw.items():
                m = PATTERN_KEY_RE.match(str(key))
                if m and int(m.group(1)) == n and isinstance(entry, dict):
                    try:
                        candidate = Grid.from_rows(entry.get("input"), key)
                    except SchemaError:
                        continue
                    if candidate.n == n:
                        pattern, filt, source = candidate, filters[n][LABEL_CROSS], "data.json"
                        break
        if pattern is None:
            pattern = generate_pattern(LABEL_CROSS, n)
            filt = generate_pattern(LABEL_CROSS, n)
        samples.append((n, pattern, filt, source))
    return samples


def run_json_mode(path: str, repeat: int) -> None:
    banner("STEP 1", "필터 로드")
    print("  파일: {}".format(os.path.abspath(path)))
    data, error = load_json(path)
    if error:
        print("  [ERR] {}".format(error))
        print("  data.json 분석을 진행할 수 없습니다. 파일 위치와 형식을 확인하세요.")
        return
    if not isinstance(data, dict):
        print("  [ERR] 최상위 구조가 객체(dict)가 아닙니다. 분석을 중단합니다.")
        return

    filters, filter_errors = load_filters(data.get("filters"))

    banner("STEP 2", "패턴 검증 · MAC 연산 · 판정 (라벨 정규화 적용)")
    patterns_raw = data.get("patterns")
    results: List[dict] = []
    if not isinstance(patterns_raw, dict):
        print("  [ERR] 'patterns' 항목이 없거나 객체(dict) 형태가 아닙니다.")
    else:
        for case_id, entry in patterns_raw.items():
            res = evaluate_case(case_id, entry, filters, filter_errors)
            results.append(res)
            print_case(res)

    banner("STEP 3", "성능 분석 (크기별 MAC 1회 평균)")
    run_benchmark(collect_bench_samples(filters, patterns_raw), repeat)

    banner("STEP 4", "결과 요약")
    print_summary(results)


def print_summary(results: List[dict]) -> None:
    total = len(results)
    passed = sum(1 for r in results if r["status"] == "PASS")
    failed = total - passed
    rate = passed / total * 100 if total else 0.0
    print("  전체 {}건  |  통과 {}건  |  실패 {}건  |  통과율 {:.1f}%".format(
        total, passed, failed, rate))

    if failed == 0:
        if total:
            print("\n  모든 케이스 통과: 라벨 정규화로 표기 차이가 제거되었고,")
            print("  epsilon({:g}) 기준으로도 동점인 케이스가 없었습니다.".format(EPSILON))
        return

    print("\n  [실패 케이스]")
    for r in results:
        if r["status"] == "FAIL":
            print("   - {:<12} {}".format(r["id"], r["reason"]))

    print("\n  [원인 분류]")
    for cat in (CAT_SCHEMA, CAT_NUMERIC, CAT_MISMATCH):
        count = sum(1 for r in results if r["category"] == cat)
        if count:
            print("   - {}: {}건".format(pad(cat, 16, "<"), count))

    ties = [r for r in results if r["category"] == CAT_NUMERIC]
    if ties:
        print("\n  [참고] epsilon 없이 단순 비교(>)했다면 동점 케이스는 이렇게 판정됩니다:")
        for r in ties:
            wrong = "오판" if r["naive"] != r["expected"] else "우연히 일치"
            print("   - {:<12} → {} ({}; 실제 차이는 부동소수점 오차 수준)".format(
                r["id"], r["naive"], wrong))
    print("\n  ※ 상세 원인 분석과 시간 복잡도 설명은 README.md '결과 리포트' 참고")


# ---------------------------------------------------------------------------
# 모드 3: 패턴 생성기 (보너스)
# ---------------------------------------------------------------------------
def run_generator_mode(repeat: int) -> None:
    banner("STEP 1", "패턴 생성기")
    while True:
        raw = ask("  생성할 크기 N (3~{} 홀수, 예: 7) > ".format(GENERATOR_MAX)).strip()
        try:
            n = int(raw)
            if n > GENERATOR_MAX:
                raise ValueError("최대 {}까지 생성할 수 있습니다.".format(GENERATOR_MAX))
            cross = generate_pattern(LABEL_CROSS, n)
            x_pat = generate_pattern(LABEL_X, n)
            break
        except ValueError as err:
            msg = str(err) if raw.lstrip("-").isdigit() else "정수를 입력하세요."
            print("  [입력 오류] {}".format(msg))

    if n <= PREVIEW_LIMIT:
        print_side_by_side([("Cross {0}x{0}".format(n), cross), ("X {0}x{0}".format(n), x_pat)])
    else:
        print("  ({0}x{0}은 너무 커서 미리보기를 생략합니다.)".format(n))

    banner("STEP 2", "생성 패턴 자기 검증 (생성 필터 사용)")
    for label, pattern in ((LABEL_CROSS, cross), (LABEL_X, x_pat)):
        cs, xs = mac_2d(pattern, cross), mac_2d(pattern, x_pat)
        verdict = verdict_from(cs, xs)
        status = "PASS" if verdict == label else "FAIL"
        print("  {:<5} 패턴 → Cross {:>6g} / X {:>6g} → 판정 {:<9} | {}".format(
            label, cs, xs, verdict, status))

    banner("STEP 3", "성능 분석 (3x3 기준 대비)")
    samples = [(3, generate_pattern(LABEL_CROSS, 3), generate_pattern(LABEL_CROSS, 3), "생성기")]
    if n != 3:
        samples.append((n, cross, cross, "생성기"))
    run_benchmark(samples, repeat)


# ---------------------------------------------------------------------------
# 진입점
# ---------------------------------------------------------------------------
def default_data_path() -> str:
    here = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data.json")
    return here if os.path.exists(here) else "data.json"


def show_menu() -> str:
    print("\n" + "#" * LINE_WIDTH)
    print("#  Mini NPU Simulator — MAC 연산으로 Cross / X 판별하기")
    print("#" * LINE_WIDTH)
    print("  1) 사용자 입력 모드   : 3x3 필터 A·B와 패턴을 직접 입력")
    print("  2) data.json 분석 모드: 5x5 / 13x13 / 25x25 일괄 판정")
    print("  3) 패턴 생성기 (보너스): N x N Cross·X 자동 생성 + 성능 측정")
    print("  0) 종료")
    return ask_choice("모드 선택 > ", ("1", "2", "3", "0"))


def main(argv: Optional[List[str]] = None) -> int:
    for stream in (sys.stdout, sys.stderr):   # 일부 콘솔의 인코딩 문제로 중단되지 않도록
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass

    parser = argparse.ArgumentParser(description="Mini NPU Simulator (MAC 연산 시뮬레이터)")
    parser.add_argument("--data", default=None, help="data.json 경로 (기본: main.py와 같은 폴더)")
    args = parser.parse_args(argv)
    data_path = args.data or default_data_path()

    try:
        choice = show_menu()
        if choice == "1":
            run_user_mode(REPEAT)
        elif choice == "2":
            run_json_mode(data_path, REPEAT)
        elif choice == "3":
            run_generator_mode(REPEAT)
    except (UserExit, KeyboardInterrupt):
        print("\n입력이 종료되어 프로그램을 마칩니다.")
        return 0
    print("\n프로그램을 종료합니다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
