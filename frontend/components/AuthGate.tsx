"use client";

import { type FormEvent, useEffect, useRef, useState } from "react";
import FlyGPTApp from "./FlyGPTApp";
import { Icon } from "./FlightIdentity";
import FlyGPTHome from "./FlyGPTHome";
import styles from "./FlyGPTHome.module.css";
import {
  getCurrentUser,
  login,
  logout,
  signup,
  type AuthUser,
} from "@/lib/auth";

export default function AuthGate() {
  const dialogRef = useRef<HTMLDialogElement>(null);
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

  function selectMode(nextMode: typeof mode) {
    setMode(nextMode);
    setError("");
  }

  function openLogin() {
    selectMode("login");
    const dialog = dialogRef.current;
    if (dialog && !dialog.open) dialog.showModal();
  }

  if (user) {
    return (
      <>
        <FlyGPTApp user={user} onSignOut={signOut} />
        {signOutError && (
          <div className="authSignoutError" role="alert">
            로그아웃 실패: {signOutError}
            <button type="button" onClick={() => setSignOutError("")} aria-label="오류 닫기">×</button>
          </div>
        )}
      </>
    );
  }

  const signupMode = mode === "signup";
  const formTitle = signupMode ? "파피티 회원가입" : "파피티 로그인";

  return (
    <FlyGPTHome onStart={openLogin}>
      <dialog ref={dialogRef} className={styles.dialog} aria-labelledby="auth-title">
        <div className={styles.dialogHead}>
          <h2 id="auth-title">{formTitle}</h2>
          <button
            className={styles.dialogClose}
            type="button"
            aria-label="로그인 창 닫기"
            onClick={() => dialogRef.current?.close()}
          >
            <Icon name="close" />
          </button>
        </div>

        {loading ? (
          <p className={styles.dialogStatus} role="status">파피티에 연결하는 중...</p>
        ) : connectionError ? (
          <div className={styles.authForm}>
            <p className={styles.dialogError} role="alert">인증 서버에 연결하지 못했어요. 연결을 확인한 뒤 다시 시도해 주세요.</p>
            <button
              className={styles.primary}
              type="button"
              onClick={() => {
                setLoading(true);
                setAuthCheckVersion((current) => current + 1);
              }}
            >
              다시 연결
            </button>
          </div>
        ) : (
          <>
            <div className={styles.dialogTabs} role="group" aria-label="인증 방식">
              <button
                type="button"
                aria-pressed={!signupMode}
                disabled={busy}
                onClick={() => selectMode("login")}
              >
                로그인
              </button>
              <button
                type="button"
                aria-pressed={signupMode}
                disabled={busy}
                onClick={() => selectMode("signup")}
              >
                회원가입
              </button>
            </div>

            <form className={styles.authForm} onSubmit={submit} aria-labelledby="auth-title">
              {signupMode && (
                <label>
                  <span>닉네임</span>
                  <input
                    value={displayName}
                    onChange={(event) => setDisplayName(event.target.value)}
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
                  minLength={8}
                  maxLength={256}
                  autoComplete={signupMode ? "new-password" : "current-password"}
                  required
                />
              </label>

              {error && <p className={styles.dialogError} role="alert">{error}</p>}

              <button className={styles.primary} type="submit" disabled={busy}>
                {busy ? "확인 중..." : signupMode ? "계정 만들고 입장" : "로그인"}
              </button>
            </form>
          </>
        )}
      </dialog>
    </FlyGPTHome>
  );
}
