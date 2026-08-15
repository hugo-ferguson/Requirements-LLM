import { Navigate, Route, Routes } from "react-router";
import { AppLayout } from "./layouts/AppLayout";
import { SessionLayout } from "./layouts/SessionLayout";
import { InputPage } from "./pages/InputPage";
import { AcReviewPage } from "./pages/AcReviewPage";
import { UatReviewPage } from "./pages/UatReviewPage";
import { EmptyStatePage } from "./pages/EmptyStatePage";
import { ComingSoonPage } from "./pages/ComingSoonPage";
import { SessionsProvider } from "./context/SessionsContext";
import { TAB_PATHS } from "./routes";

export default function App() {
  return (
    <SessionsProvider>
      <Routes>
        <Route element={<AppLayout />}>
          <Route index element={<EmptyStatePage />} />
          <Route path="sessions/:sessionId" element={<SessionLayout />}>
            <Route index element={<Navigate to={TAB_PATHS.input} replace />} />
            <Route path={TAB_PATHS.input} element={<InputPage />} />
            <Route path={TAB_PATHS.acReview} element={<AcReviewPage />} />
            <Route path={TAB_PATHS.uatReview} element={<UatReviewPage />} />
            <Route path={TAB_PATHS.export} element={<ComingSoonPage title="Export" />} />
          </Route>
          <Route path="*" element={<Navigate to="/" replace />} />
        </Route>
      </Routes>
    </SessionsProvider>
  );
}
