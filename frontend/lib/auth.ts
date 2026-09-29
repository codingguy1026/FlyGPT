export type AuthUser = {
  id: string;
  email: string;
  display_name: string;
  created_at: number;
};

type AuthResponse = {
  user: AuthUser;
};

async function authRequest<T>(
  input: string,
  init?: RequestInit,
): Promise<T> {
  const response = await fetch(input, {
    ...init,
    credentials: "same-origin",
  });

  let payload: unknown = null;
  try {
    payload = await response.json();
  } catch {
    // handled below
  }

  if (!response.ok) {
    const detail =
      typeof payload === "object" &&
      payload !== null &&
      "detail" in payload
        ? String((payload as { detail: unknown }).detail)
        : `HTTP ${response.status}`;
    throw new Error(detail);
  }

  return payload as T;
}

export async function getCurrentUser(): Promise<AuthUser | null> {
  const response = await fetch("/api/auth/me", {
    cache: "no-store",
    credentials: "same-origin",
  });

  if (response.status === 401) return null;
  if (!response.ok) throw new Error(`HTTP ${response.status}`);

  const payload = (await response.json()) as AuthResponse;
  return payload.user;
}

export async function login(
  email: string,
  password: string,
): Promise<AuthUser> {
  const payload = await authRequest<AuthResponse>("/api/auth/login", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ email, password }),
  });
  return payload.user;
}

export async function signup(
  email: string,
  password: string,
  displayName: string,
): Promise<AuthUser> {
  const payload = await authRequest<AuthResponse>("/api/auth/signup", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      email,
      password,
      display_name: displayName,
    }),
  });
  return payload.user;
}

export async function logout(): Promise<void> {
  await authRequest<{ ok: boolean }>("/api/auth/logout", {
    method: "POST",
  });
}
