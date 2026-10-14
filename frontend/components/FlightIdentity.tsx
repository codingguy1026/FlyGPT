"use client";

type IconName =
  | "home"
  | "plus"
  | "brain"
  | "palette"
  | "settings"
  | "copy"
  | "thumb"
  | "retry"
  | "spark"
  | "code"
  | "search"
  | "send"
  | "menu"
  | "close";

export function Icon({ name, size = 20 }: { name: IconName; size?: number }) {
  const common = {
    width: size,
    height: size,
    viewBox: "0 0 24 24",
    fill: "none",
    stroke: "currentColor",
    strokeWidth: 1.8,
    strokeLinecap: "round" as const,
    strokeLinejoin: "round" as const,
    "aria-hidden": true,
  };

  if (name === "home") return <svg {...common}><path d="M4 10.5 12 4l8 6.5v8.5a1 1 0 0 1-1 1h-5v-6h-4v6H5a1 1 0 0 1-1-1z" /></svg>;
  if (name === "plus") return <svg {...common}><path d="M12 5v14M5 12h14" /></svg>;
  if (name === "brain") return <svg {...common}><path d="M9 4.5A3.5 3.5 0 0 0 5.5 8v1A3.5 3.5 0 0 0 4 15.7 3.5 3.5 0 0 0 9.5 19H11V5.5A2 2 0 0 0 9 3.5zM15 4.5A3.5 3.5 0 0 1 18.5 8v1a3.5 3.5 0 0 1 1.5 6.7A3.5 3.5 0 0 1 14.5 19H13V5.5a2 2 0 0 1 2-2z" /><path d="M7 10h4M13 13h4M8 16h3M13 8h4" /></svg>;
  if (name === "palette") return <svg {...common}><path d="M12 3a9 9 0 1 0 0 18h1.5a2 2 0 0 0 0-4H12a2 2 0 0 1 0-4h4a5 5 0 0 0 0-10z" /><circle cx="7.5" cy="9" r="1" /><circle cx="9.5" cy="6.5" r="1" /><circle cx="13" cy="6.2" r="1" /></svg>;
  if (name === "settings") return <svg {...common}><circle cx="12" cy="12" r="3" /><path d="M19 13.5v-3l-2-.7a7 7 0 0 0-.7-1.7l.9-1.9-2.1-2.1-1.9.9a7 7 0 0 0-1.7-.7L10.5 2h-3l-.7 2a7 7 0 0 0-1.7.7l-1.9-.9-2.1 2.1.9 1.9a7 7 0 0 0-.7 1.7L0 10.5v3l2 .7a7 7 0 0 0 .7 1.7l-.9 1.9 2.1 2.1 1.9-.9a7 7 0 0 0 1.7.7l.7 2h3l.7-2a7 7 0 0 0 1.7-.7l1.9.9 2.1-2.1-.9-1.9a7 7 0 0 0 .7-1.7z" /></svg>;
  if (name === "copy") return <svg {...common}><rect x="8" y="8" width="11" height="11" rx="2" /><path d="M16 8V6a2 2 0 0 0-2-2H6a2 2 0 0 0-2 2v8a2 2 0 0 0 2 2h2" /></svg>;
  if (name === "thumb") return <svg {...common}><path d="M7 21H4V10h3zM7 11l4-8 2 .5a2 2 0 0 1 1.3 2.5L13 10h5.5a2 2 0 0 1 2 2.3l-1 6A3 3 0 0 1 16.5 21H7z" /></svg>;
  if (name === "retry") return <svg {...common}><path d="M20 7v5h-5" /><path d="M18.5 16a8 8 0 1 1 .7-7L20 12" /></svg>;
  if (name === "spark") return <svg {...common}><path d="m12 3 1.2 3.8L17 8l-3.8 1.2L12 13l-1.2-3.8L7 8l3.8-1.2zM5 14l.8 2.2L8 17l-2.2.8L5 20l-.8-2.2L2 17l2.2-.8zM19 14l.7 1.8 1.8.7-1.8.7L19 19l-.7-1.8-1.8-.7 1.8-.7z" /></svg>;
  if (name === "code") return <svg {...common}><path d="m9 7-5 5 5 5M15 7l5 5-5 5M13.5 4l-3 16" /></svg>;
  if (name === "search") return <svg {...common}><circle cx="10.5" cy="10.5" r="6.5" /><path d="m15.5 15.5 5 5" /></svg>;
  if (name === "send") return <svg {...common}><path d="M12 19V5M6.5 10.5 12 5l5.5 5.5" /></svg>;
  if (name === "menu") return <svg {...common}><path d="M4 7h16M4 12h16M4 17h16" /></svg>;
  if (name === "close") return <svg {...common}><path d="m6 6 12 12M18 6 6 18" /></svg>;
  return null;
}

export function FlyOrb({ small = false }: { small?: boolean }) {
  return (
    <div className={small ? "flyOrb small" : "flyOrb"} aria-hidden="true">
      <span className="orbRing ringOne" />
      <span className="orbRing ringTwo" />
      <span className="flyCore">
        <span className="flyWing leftWing" />
        <span className="flyWing rightWing" />
        <span className="flyBody">FLY</span>
      </span>
    </div>
  );
}

