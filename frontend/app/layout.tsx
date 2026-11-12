import Script from "next/script";
import type { Metadata } from "next";
import "./globals.css";
import "./refinements.css";
import "./liquid-refraction.css";
import "./studio-theme.css";
import "./neuron-explorer.css";
import "./papiti-orbit.css";

export const metadata: Metadata = {
  title: "papiti. | 생각이 움직이는 공간",
  description: "파피티, 호기심이 생각으로 연결되는 AI 대화 공간",
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
