"use client";

import { type ReactNode, useState } from "react";
import ConnectomeSpecimen from "./ConnectomeSpecimen";
import { Icon } from "./FlightIdentity";
import styles from "./FlyGPTHome.module.css";

const repository = "https://github.com/codingguy1026/FlyGPT";

const stages = [
  {
    icon: "spark",
    label: "질문 이해",
    title: "하나의 질문에서 시작해요.",
    description: "질문을 입력하면 파피티가 대화 맥락을 살펴보고 어떤 처리가 필요한지 판단해요.",
  },
  {
    icon: "brain",
    label: "커넥톰 라우팅",
    title: "뉴런 연결에서 처리 경로를 찾아요.",
    description: "파피티의 실험용 라우터는 MaleCNS 연결 구조를 활용해 질문을 처리할 경로를 골라요.",
  },
  {
    icon: "code",
    label: "응답 생성",
    title: "선택한 경로를 따라 답변을 만들어요.",
    description: "필요한 도구나 언어 모델이 답변 생성을 맡아요. 채팅의 뉴런 패널에서 라우팅 과정을 살펴볼 수 있어요.",
  },
] as const;

export default function FlyGPTHome({
  onStart,
  children,
}: {
  onStart: () => void;
  children: ReactNode;
}) {
  const [stage, setStage] = useState(0);

  return (
    <div className={styles.home}>
      <header className={styles.header}>
        <div className={styles.headerInner}>
          <a href="#" className={styles.brand} aria-label="FlyGPT 홈">
            <img src="/papiti-mark.svg" alt="" width={36} height={36} />
            <span>Fly<span>GPT</span></span>
          </a>
          <nav className={styles.nav} aria-label="홈페이지 탐색">
            <a href="#connectome">커넥톰</a>
            <a href="#research">작동 방식</a>
            <a href={repository}>GitHub</a>
          </nav>
          <button className={styles.headerButton} type="button" onClick={onStart}>로그인</button>
        </div>
      </header>

      <main className={styles.wrap}>
        <section className={styles.hero} aria-labelledby="home-title">
          <div className={styles.heroCopy}>
            <p className={styles.eyebrow}><span /> 커넥톰 기반 AI 연구 프로젝트</p>
            <h1 id="home-title">초파리의 뇌에서<br />시작된 <em>AI.</em></h1>
            <p className={styles.intro}>
              파피티는 초파리의 뉴런 연결망을 이용해 질문의 처리 경로를 찾는 실험적인 대화형 AI예요.
            </p>
            <div className={styles.actions}>
              <button className={styles.primary} type="button" onClick={onStart}>
                <Icon name="spark" size={19} /> 파피티 시작하기
              </button>
              <a className={styles.secondary} href="#research">작동 방식 보기</a>
            </div>
          </div>

          <ConnectomeSpecimen />
        </section>

        <div className={styles.facts}>
          <div><strong>166,700</strong><span>프로젝트의 뉴런 규모</span></div>
          <div><strong>MaleCNS</strong><span>수컷 초파리 중추신경계 데이터</span></div>
          <div><strong>Hybrid</strong><span>커넥톰 라우터와 응답 모델</span></div>
        </div>

        <section id="research" className={styles.research} aria-labelledby="research-title">
          <div className={styles.researchHeading}>
            <p className={styles.eyebrow}>파피티의 작동 방식</p>
            <h2 id="research-title">질문에서 답변까지</h2>
          </div>
          <div className={styles.stages} role="group" aria-label="응답 처리 단계">
            {stages.map(({ icon, label }, index) => (
              <button
                key={label}
                type="button"
                aria-pressed={stage === index}
                aria-controls="research-stage"
                onClick={() => setStage(index)}
              >
                <Icon name={icon} size={21} />
                <span>{label}</span>
              </button>
            ))}
          </div>
          <div id="research-stage" className={styles.stageDetail} aria-live="polite" aria-atomic="true">
            <h3>{stages[stage].title}</h3>
            <p>{stages[stage].description}</p>
          </div>
        </section>

        <footer className={styles.footer}>
          <span>FlyGPT <span>·</span> 파피티</span>
          <a href={repository}>GitHub에서 소스 보기</a>
        </footer>
      </main>
      {children}
    </div>
  );
}
