package exceptions;

/**
 * Created by davidetalon on 28/07/17.
 */
public class InsufficientVideoRealTracesException extends Exception{

    public InsufficientVideoRealTracesException() {
        super();
    }

    public InsufficientVideoRealTracesException(String message) {
        super(message);
    }

    public InsufficientVideoRealTracesException(String message, Throwable cause) {
        super(message, cause);
    }

    public InsufficientVideoRealTracesException(Throwable cause) {
        super(cause);
    }

}
