type ToastProps = {
    show: boolean;
    message: string;
}

export const Toast = ({ show, message }: ToastProps) => {
    if (!show) return null;

    return (
        <div className="fixed top-4 right-4 bg-[#2A1515] text-[#D4A6A6] px-6 py-3 rounded-lg shadow-lg transition-opacity duration-300">
            {message}
        </div>
    );
};
