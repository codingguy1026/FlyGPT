import Script from "next/script";
import type { Metadata } from "next";
import "./globals.css";
import "./refinements.css";
import "./liquid-refraction.css";
import "./studio-theme.css";
import "./neuron-explorer.css";
import "./papiti-orbit.css";
import "./malecns-observatory.css";

export const metadata: Metadata = {
  title: "FlyGPT | 초파리의 뇌에서 시작된 AI",
  description: "FlyGPT(파피티)는 초파리의 뉴런 연결망을 활용하는 실험적인 대화형 AI 프로젝트입니다.",
  icons: {
    icon: "/papiti-mark.svg",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="ko">
      <body>
        {children}
        <Script src="/liquid-refraction.js" strategy="afterInteractive" />
      </body>
    </html>
  );
}
