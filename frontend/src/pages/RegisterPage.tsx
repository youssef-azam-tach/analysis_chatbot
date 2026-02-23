import { useState } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import { useAuthStore } from '@/stores/auth-store';
import { Button, Input } from '@/components/ui';
import { UserPlus } from 'lucide-react';
import toast from 'react-hot-toast';

export default function RegisterPage() {
  const [email, setEmail] = useState('');
  const [fullName, setFullName] = useState('');
  const [password, setPassword] = useState('');
  const [confirm, setConfirm] = useState('');
  const [loading, setLoading] = useState(false);
  const register = useAuthStore((s) => s.register);
  const navigate = useNavigate();

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (password !== confirm) { toast.error('Passwords don\'t match'); return; }
    setLoading(true);
    try {
      await register(email, fullName, password);
      toast.success('Account created! Please sign in.');
      navigate('/login');
    } catch (err: any) {
      toast.error(err.response?.data?.detail || 'Registration failed');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center p-8 bg-[var(--bg-primary)]">
      <div className="w-full max-w-md">
        <div className="flex items-center gap-3 mb-10">
          <div className="w-10 h-10 rounded-xl gradient-accent flex items-center justify-center text-white font-bold">AI</div>
          <span className="text-xl font-bold gradient-text">DataAnalysis</span>
        </div>

        <h2 className="text-2xl font-bold mb-2">Create your account</h2>
        <p className="text-[var(--text-secondary)] mb-8">Start analyzing your data with AI-powered insights</p>

        <form onSubmit={handleSubmit} className="space-y-5">
          <Input label="Full Name" placeholder="John Doe" value={fullName} onChange={setFullName} required />
          <Input label="Email" type="email" placeholder="you@company.com" value={email} onChange={setEmail} required />
          <Input label="Password" type="password" placeholder="••••••••" value={password} onChange={setPassword} required />
          <Input label="Confirm Password" type="password" placeholder="••••••••" value={confirm} onChange={setConfirm} required />
          <Button type="submit" loading={loading} className="w-full" size="lg" icon={<UserPlus className="w-4 h-4" />}>Create Account</Button>
        </form>

        <p className="text-center text-sm text-[var(--text-muted)] mt-8">
          Already have an account?{' '}
          <Link to="/login" className="text-[var(--accent)] hover:text-[var(--accent-hover)] font-medium">Sign in</Link>
        </p>
      </div>
    </div>
  );
}
