import { Outlet, useParams } from "react-router";
import { TabBar } from "./TabBar";

export function SessionLayout() {
  const { sessionId } = useParams<{ sessionId: string }>();

  if (!sessionId) return null;

  return (
    <div className="flex h-full flex-col">
      <TabBar sessionId={sessionId} />
      <div className="min-h-0 flex-1">
        <Outlet />
      </div>
    </div>
  );
}
