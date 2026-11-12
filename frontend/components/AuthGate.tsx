"use client";

import { type FormEvent, useEffect, useState } from "react";
import FlyGPTApp from "./FlyGPTApp";
import { PapitiMark } from "./FlightIdentity";
import {
  getCurrentUser,
  login,
  logout,
  signup,
  type AuthUser,
} from "@/lib/auth";

export default function AuthGate() {
  const [user, setUser] = useState<AuthUser | null>(null);
  const [loading, setLoading] = useState(true);
  const [connectionError, setConnectionError] = useState(false);
  const [authCheckVersion, setAuthCheckVersion] = useState(0);
  const [busy, setBusy] = useState(false);
  const [mode, setMode] = useState<"login" | "signup">("login");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [displayName, setDisplayName] = useState("");
  const [error, setError] = useState("");
  const [signOutError, setSignOutError] = useState("");

  useEffect(() => {
    let active = true;
    setLoading(true);
    setConnectionError(false);

    getCurrentUser()
      .then((nextUser) => {
        if (active) setUser(nextUser);
      })
      .catch(() => {
        // A network/server failure is not the same as an unauthenticated 401.
        if (active) setConnectionError(true);
      })
      .finally(() => {
        if (active) setLoading(false);
      });

    return () => {
      active = false;
    };
  }, [authCheckVersion]);

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (busy) return;

    setBusy(true);
    setError("");

    try {
      const nextUser =
        mode === "signup"
          ? await signup(email, password, displayName)
          : await login(email, password);
      setUser(nextUser);
      setPassword("");
    } catch (reason: unknown) {
      setError(reason instanceof Error ? reason.message : String(reason));
    } finally {
      setBusy(false);
    }
  }

  async function signOut() {
    setSignOutError("");
    try {
      await logout();
      setUser(null);
      setPassword("");
    } catch (reason: unknown) {
      setSignOutError(reason instanceof Error ? reason.message : "로그아웃에 실패했습니다. 다시 시도해 주세요.");
    }
  }

  if (loading) {
    return (
      <main className="authShell">
        <div className="authLoading">
          <PapitiMark />
          <span>파피티에 연결하는 중...</span>
        </div>
      </main>
    );
  }

  if (connectionError) {
    return (
      <main className="authShell">
        <section className="authCard">
          <div className="authBrand">
            <PapitiMark />
            <div>
              <span>CONNECTION CHECK</span>
              <h1>papiti.</h1>
            </div>
          </div>
          <div className="authIntro">
            <span>CONNECTION UNAVAILABLE</span>
            <h2>서버 상태를 확인하지 못했어요.</h2>
            <p>로그아웃된 것은 아닐 수 있어요. 연결을 확인한 뒤 다시 시도해 주세요.</p>
          </div>
          <div className="authError" role="alert">
            인증 서버에 연결하지 못했습니다. 개발 환경이라면 터미널에서 make doctor로
            프론트엔드와 백엔드 상태를 확인할 수 있어요.
          </div>
          <button
            className="authSubmit authRetryButton"
            type="button"
            onClick={() => setAuthCheckVersion((current) => current + 1)}
          >
            다시 연결
          </button>
        </section>
      </main>
    );
  }

  if (user) {
    return <>
      <FlyGPTApp user={user} onSignOut={signOut} />
      {signOutError && <div className="authSignoutError" role="alert">로그아웃 실패: {signOutError}<button type="button" onClick={() => setSignOutError("")} aria-label="오류 닫기">×</button></div>}
    </>;
  }

  return (
    <main className="authShell">
      <section className="authCard">
        <div className="authBrand">
          <PapitiMark />
          <div>
            <span>CURIOSITY IN MOTION</span>
            <h1>papiti.</h1>
          </div>
        </div>

        <div className="authIntro">
          <span>{mode === "login" ? "WELCOME BACK" : "JOIN PAPITI"}</span>
          <h2>{mode === "login" ? "다시 만나서 반가워!" : "파피티와 생각을 시작해 봐."}</h2>
          <p>
            {mode === "login"
              ? "로그인하면 이전 대화를 이어가고, 파피티의 다양한 기능을 이용할 수 있어."
              : "계정마다 대화 기록과 기억을 분리해서 관리해."}
          </p>
        </div>

        <div className="authTabs" role="tablist" aria-label="인증 방식">
          <button
            className={mode === "login" ? "active" : ""}
            role="tab"
            aria-selected={mode === "login"}
            type="button"
            onClick={() => { setMode("login"); setError(""); }}
          >
            로그인
          </button>
          <button
            className={mode === "signup" ? "active" : ""}
            role="tab"
            aria-selected={mode === "signup"}
            type="button"
            onClick={() => { setMode("signup"); setError(""); }}
          >
            회원가입
          </button>
        </div>

        <form className="authForm" onSubmit={submit} aria-label={mode === "login" ? "FlyGPT 로그인" : "FlyGPT 회원가입"}>
          {mode === "signup" && (
            <label>
              <span>닉네임</span>
              <input
                value={displayName}
                onChange={(event) => setDisplayName(event.target.value)}
                placeholder="파피티 조련사"
                maxLength={50}
                autoComplete="nickname"
              />
            </label>
          )}

          <label>
            <span>이메일</span>
            <input
              value={email}
              onChange={(event) => setEmail(event.target.value)}
              type="email"
              placeholder="you@example.com"
              autoComplete="email"
              required
            />
          </label>

          <label>
            <span>비밀번호</span>
            <input
              value={password}
              onChange={(event) => setPassword(event.target.value)}
              type="password"
              placeholder="8자 이상"
              minLength={8}
              maxLength={256}
              autoComplete={mode === "signup" ? "new-password" : "current-password"}
              required
            />
          </label>

          {error && <div className="authError" role="alert">{error}</div>}

          <button className="authSubmit" type="submit" disabled={busy}>
            {busy
              ? "확인 중..."
              : mode === "login"
                ? "로그인"
                : "계정 만들고 입장"}
          </button>
        </form>

        <p className="authFootnote">
          비밀번호는 원문으로 저장하지 않고 해시 처리됩니다.
        </p>
      </section>
    </main>
  );
}
