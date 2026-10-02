"use client";

import { type FormEvent, useEffect, useState } from "react";
import FlyGPTApp from "./FlyGPTApp";
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
  const [busy, setBusy] = useState(false);
  const [mode, setMode] = useState<"login" | "signup">("login");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [displayName, setDisplayName] = useState("");
  const [error, setError] = useState("");

  useEffect(() => {
    getCurrentUser()
      .then(setUser)
      .catch(() => setUser(null))
      .finally(() => setLoading(false));
  }, []);

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
    try {
      await logout();
    } finally {
      setUser(null);
      setPassword("");
    }
  }

  if (loading) {
    return (
      <main className="authShell">
        <div className="authLoading">
          <img src="/flygpt-logo.webp" alt="" />
          <span>FlyGPT 출입증 확인 중...</span>
        </div>
      </main>
    );
  }

  if (user) {
    return <FlyGPTApp user={user} onSignOut={signOut} />;
  }

  return (
    <main className="authShell">
      <section className="authCard">
        <div className="authBrand">
          <img src="/flygpt-logo.webp" alt="FlyGPT logo" />
          <div>
            <span>CONNECTOME AI</span>
            <h1>FlyGPT</h1>
          </div>
        </div>

        <div className="authIntro">
          <span>{mode === "login" ? "WELCOME BACK" : "CREATE FLIGHT ID"}</span>
          <h2>{mode === "login" ? "다시 비행할 시간." : "파피티 계정을 만들자."}</h2>
          <p>
            {mode === "login"
              ? "로그인하면 네 FlyGPT 작업공간으로 들어갑니다."
              : "계정마다 채팅 기록과 FlyGPT 메모리를 따로 보관합니다."}
          </p>
        </div>

        <div className="authTabs" role="tablist" aria-label="인증 방식">
          <button
            className={mode === "login" ? "active" : ""}
            type="button"
            onClick={() => { setMode("login"); setError(""); }}
          >
            로그인
          </button>
          <button
            className={mode === "signup" ? "active" : ""}
            type="button"
            onClick={() => { setMode("signup"); setError(""); }}
          >
            회원가입
          </button>
        </div>

        <form className="authForm" onSubmit={submit}>
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

          {error && <div className="authError">{error}</div>}

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
