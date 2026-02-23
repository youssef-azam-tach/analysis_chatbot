import { useState } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import { useAuthStore } from '@/stores/auth-store';
import { Button, Input } from '@/components/ui';
import { Brain, ArrowRight } from 'lucide-react';
import toast from 'react-hot-toast';

export default function LoginPage() {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [loading, setLoading] = useState(false);
  const login = useAuthStore((s) => s.login);
  const navigate = useNavigate();

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    try {
      await login(email, password);
      toast.success('Welcome back!');
      navigate('/app');
    } catch (err: any) {
      toast.error(err.response?.data?.detail || 'Login failed');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex">
      {/* Left — Branding */}
      <div className="hidden lg:flex w-1/2 gradient-accent flex-col items-center justify-center p-12 relative overflow-hidden">
        <div className="absolute inset-0 bg-[url('data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iNjAiIGhlaWdodD0iNjAiIHZpZXdCb3g9IjAgMCA2MCA2MCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48ZyBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPjxnIGZpbGw9IiNmZmYiIGZpbGwtb3BhY2l0eT0iMC4wNSI+PHBhdGggZD0iTTM2IDM0djItSDI0di0yaDEyem0wLTRWMjhIMjR2MmgxMnptLTIwIDBoMnYyMGgtMlYzMHoiLz48L2c+PC9nPjwvc3ZnPg==')] opacity-30" />
        <div className="relative z-10 text-center">
          <div className="w-20 h-20 rounded-2xl bg-white/10 backdrop-blur flex items-center justify-center mx-auto mb-8">
            <Brain className="w-10 h-10 text-white" />
          </div>
          <h1 className="text-4xl font-bold text-white mb-4">AI Data Analysis</h1>
          <p className="text-lg text-white/80 max-w-md">Transform raw data into verified, actionable business intelligence with AI-powered analytics.</p>
          <div className="flex gap-6 mt-12 text-sm text-white/70">
            <div className="text-center"><p className="text-3xl font-bold text-white">101</p><p>API Endpoints</p></div>
            <div className="text-center"><p className="text-3xl font-bold text-white">13</p><p>Analysis Pages</p></div>
            <div className="text-center"><p className="text-3xl font-bold text-white">∞</p><p>Insights</p></div>
          </div>
        </div>
      </div>

      {/* Right — Form */}
      <div className="flex-1 flex items-center justify-center p-8 bg-[var(--bg-primary)]">
        <div className="w-full max-w-md">
          <div className="lg:hidden flex items-center gap-3 mb-10">
            <div className="w-10 h-10 rounded-xl gradient-accent flex items-center justify-center text-white font-bold">AI</div>
            <span className="text-xl font-bold gradient-text">DataAnalysis</span>
          </div>

          <h2 className="text-2xl font-bold mb-2">Sign in to your account</h2>
          <p className="text-[var(--text-secondary)] mb-8">Enter your credentials to access the platform</p>

          <form onSubmit={handleSubmit} className="space-y-5">
            <Input label="Email" type="email" placeholder="you@company.com" value={email} onChange={setEmail} required />
            <Input label="Password" type="password" placeholder="••••••••" value={password} onChange={setPassword} required />
            <Button type="submit" loading={loading} className="w-full" size="lg" icon={<ArrowRight className="w-4 h-4" />}>Sign in</Button>
          </form>

          <p className="text-center text-sm text-[var(--text-muted)] mt-8">
            Don't have an account?{' '}
            <Link to="/register" className="text-[var(--accent)] hover:text-[var(--accent-hover)] font-medium">Create account</Link>
          </p>
        </div>
      </div>
    </div>
  );
}
