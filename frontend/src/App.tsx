import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import AppLayout from '@/layouts/AppLayout';
import LoginPage from '@/pages/LoginPage';
import RegisterPage from '@/pages/RegisterPage';
import HomePage from '@/pages/HomePage';
import UploadPage from '@/pages/UploadPage';
import SchemaPage from '@/pages/SchemaPage';
import QualityPage from '@/pages/QualityPage';
import CleaningPage from '@/pages/CleaningPage';
import GoalsPage from '@/pages/GoalsPage';
import AnalysisPage from '@/pages/AnalysisPage';
import KpisPage from '@/pages/KpisPage';
import VisualizationPage from '@/pages/VisualizationPage';
import DashboardPage from '@/pages/DashboardPage';
import ChatPage from '@/pages/ChatPage';
import ReportsPage from '@/pages/ReportsPage';
import QuickExcelPage from '@/pages/QuickExcelPage';

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        {/* Public routes */}
        <Route path="/login" element={<LoginPage />} />
        <Route path="/register" element={<RegisterPage />} />

        {/* Protected routes */}
        <Route element={<AppLayout />}>
          <Route path="/" element={<HomePage />} />
          <Route path="/upload" element={<UploadPage />} />
          <Route path="/schema" element={<SchemaPage />} />
          <Route path="/quality" element={<QualityPage />} />
          <Route path="/cleaning" element={<CleaningPage />} />
          <Route path="/goals" element={<GoalsPage />} />
          <Route path="/analysis" element={<AnalysisPage />} />
          <Route path="/kpis" element={<KpisPage />} />
          <Route path="/visualization" element={<VisualizationPage />} />
          <Route path="/dashboard" element={<DashboardPage />} />
          <Route path="/chat" element={<ChatPage />} />
          <Route path="/reports" element={<ReportsPage />} />
          <Route path="/quick-excel" element={<QuickExcelPage />} />
        </Route>

        {/* Catch all */}
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </BrowserRouter>
  );
}
