import { Navigate, Route, Routes } from "react-router";
import { AppLayout } from "./layouts/AppLayout";
import { InputPage } from "./pages/InputPage";
import { ComingSoonPage } from "./pages/ComingSoonPage";
import { ROUTES } from "./routes";

export default function App() {
  return (
    <Routes>
      <Route element={<AppLayout />}>
        <Route index element={<Navigate to={ROUTES.input} replace />} />
        <Route path={ROUTES.input} element={<InputPage />} />
        <Route path={ROUTES.acReview} element={<ComingSoonPage title="AC Review" />} />
        <Route path={ROUTES.uatReview} element={<ComingSoonPage title="UAT Review" />} />
        <Route path={ROUTES.export} element={<ComingSoonPage title="Export" />} />
        <Route path="*" element={<Navigate to={ROUTES.input} replace />} />
      </Route>
    </Routes>
  );
}
