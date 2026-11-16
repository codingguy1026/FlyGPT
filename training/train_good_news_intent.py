"""Train a binary classifier on labeled Korean turns without external dependencies."""

import json
import math
from pathlib import Path
import random

from good_news_intent import features

ANNOUNCEMENTS = (
    "나한테 기쁜 소식이 있어", "오늘 완전 신나는 일이 있었어", "오늘 좋은 일이 생겼어",
    "반가운 소식이 들어왔어", "기분 좋은 소식을 들었어요", "오늘은 좋은 일이 있었어",
    "드디어 기다리던 좋은 소식이 왔어", "좋은 일이 생겼어요", "좋은 소식이 하나 있어",
    "드디어 합격했어", "시험에 붙었어", "오늘 상을 받았어", "오늘 목표를 달성했어",
    "좋은 일이 있어서 기분이 좋아", "좋은 일이 생겨서 신나", "기뻐서 얘기하고 싶어",
    "결과가 잘 나왔어", "나 드디어 해냈어", "오늘 정말 좋은 일이 있었어요",
    "기쁜 일이 있었어요", "내게 아주 반가운 소식이 왔어", "좋은 소식부터 말할게",
    "드디어 프로젝트가 성공했어", "나 기쁜 소식 전하러 왔어", "아까 좋은 일 있었지",
    "방금 기쁜 소식 들었어", "좋은 소식이 왔습니다", "경기에서 이겼어",
)
OTHER_TURNS = (
    "오늘 좋은 뉴스 좀 찾아줘", "기쁜 소식만 골라서 검색해 줘", "좋은 소식 있나 알아봐",
    "좋은 일 있었어?", "오늘 좋은 소식이 있을까", "어떤 좋은 일이 있을지 알려줘",
    "오늘 완전 피곤했어", "오늘은 아무 일도 없었어", "나쁜 일이 생겼어",
    "반갑지 않은 소식을 들었어", "좋은 일이 생길 것 같아", "기쁜 소식은 없었어",
    "좋은 소식이 아니야", "오늘은 나쁜 소식이 있어", "요즘 나쁜 일만 생겼어",
    "좋은 일이라는 표현을 설명해 줘", "최신 좋은 뉴스를 모아서 보여 줘", "좋은 날씨였어",
    "요즘 기분이 좋지 않아", "오늘 일 있었어", "내일 좋은 일이 있으면 좋겠어",
    "좋은 소식이 없다는 소식을 들었어", "좋은 소식을 기다리고 있어",
    "이 영화 제목이 좋은 소식이야", "오늘 뉴스를 봤어", "결과가 아직 안 나왔어",
    "서울 오늘 날씨", "칠 곱하기 육", "내가 지난번에 뭐라고 했지", "왜 그런 일이 생겼어",
    "합격 소식을 보고 싶어", "다른 사람이 좋은 일을 했대", "좋은 소식이 있는지 확인해 줘",
    "네가 좋은 소식을 상상해서 써 줘", "좋은 일이 전혀 없었어", "실시간 뉴스",
    "새로운 일이 생겼어", "성공할 수 있을 것 같아",
)


def train() -> dict:
    examples = [(sorted(features(text)), label)
                for group, label in ((ANNOUNCEMENTS, 1), (OTHER_TURNS, 0))
                for text in group]
    weights = {}
    bias = 0.0
    rng = random.Random(42)
    for _ in range(160):
        rng.shuffle(examples)
        for terms, label in examples:
            score = bias + sum(weights.get(term, 0) for term in terms)
            delta = 0.12 * (label - 1 / (1 + math.exp(-score)))
            bias += delta
            for term in terms:
                weights[term] = 0.9995 * weights.get(term, 0) + delta
    return {"bias": round(bias, 4), "threshold": 0.3,
            "weights": {term: round(weight, 4) for term, weight in sorted(weights.items())}}


if __name__ == "__main__":
    Path(__file__).with_name("good_news_intent_weights.json").write_text(
        json.dumps(train(), ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8"
    )
