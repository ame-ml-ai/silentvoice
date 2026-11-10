import { lazy, Suspense } from 'react'
import { BrowserRouter, Link, Route, Routes } from 'react-router-dom'
import Layout from './components/Layout'
import { Skeleton } from './components/ui'
import HomePage from './pages/HomePage'
import MethodologyPage from './pages/MethodologyPage'
import ModelPage from './pages/ModelPage'
import RecognizePage from './pages/RecognizePage'

// charts (recharts) are only needed on the research page
const ResearchPage = lazy(() => import('./pages/ResearchPage'))

function NotFound() {
  return (
    <div className="py-20 text-center">
      <p className="font-mono text-sm text-fg-subtle">404</p>
      <p className="mt-2 text-fg">This page does not exist.</p>
      <Link to="/" className="mt-4 inline-block text-sm text-accent hover:underline">Back to overview</Link>
    </div>
  )
}

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route element={<Layout />}>
          <Route index element={<HomePage />} />
          <Route path="recognize" element={<RecognizePage />} />
          <Route path="research" element={<Suspense fallback={<Skeleton className="h-96" />}><ResearchPage /></Suspense>} />
          <Route path="model" element={<ModelPage />} />
          <Route path="methodology" element={<MethodologyPage />} />
          <Route path="*" element={<NotFound />} />
        </Route>
      </Routes>
    </BrowserRouter>
  )
}
