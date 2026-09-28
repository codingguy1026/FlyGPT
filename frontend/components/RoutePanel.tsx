import type { RouterResult } from "@/lib/api";

type Props = {
  router: RouterResult | null;
};

export default function RoutePanel({ router }: Props) {
  return (
    <section className="routeCard">
      <div className="routeCardTop">
        <div>
          <div className="eyebrow">ROUTE DECISION</div>
          <div className="routeName">
            {router?.route ?? "Waiting for a question"}
          </div>
        </div>
        <div className="routeConfidence">
          {router ? `${(router.confidence * 100).toFixed(1)}%` : "—"}
        </div>
      </div>

      <div className="routeBars">
        {(router?.top_routes ?? []).length === 0 ? (
          <div className="routePlaceholder">
            질문을 보내면 라우팅 결과가 표시됩니다.
          </div>
        ) : (
          router?.top_routes?.map((item) => (
            <div className="routeRow" key={item.route}>
              <div className="routeRowHead">
                <span>{item.route}</span>
                <span>{(item.confidence * 100).toFixed(1)}%</span>
              </div>
              <div className="routeTrack">
                <div
                  className="routeFill"
                  style={{
                    width: `${Math.max(2, item.confidence * 100)}%`,
                  }}
                />
              </div>
            </div>
          ))
        )}
      </div>
    </section>
  );
}
